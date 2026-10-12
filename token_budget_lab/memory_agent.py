"""Python agent dispatch with metered recovery and execution gates."""
from __future__ import annotations
import copy
import json
import os
import time
import urllib.error
import urllib.request

from .deepseek import ENDPOINT, NoRedirect, normalize_usage
from .memory import PreparedHistory, encode, scope_key, validate_messages

class DeepSeekChat:
    def __init__(self, model='deepseek-flash', max_tokens=1024, timeout=120, json_output=False):
        self.key = os.environ.get('DEEPSEEK_API_KEY')
        if not self.key:
            raise ValueError('set DEEPSEEK_API_KEY locally')
        if not model or max_tokens < 1 or timeout <= 0:
            raise ValueError('invalid model/output/timeout')
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        self.json_output = json_output
        self.opener = urllib.request.build_opener(NoRedirect())
        self.last = {}

    def __call__(self, messages, tools):
        self.last = {}
        body = dict(model=self.model, messages=messages,
                    thinking=dict(type='disabled'), stream=False, max_tokens=self.max_tokens)
        if tools:
            body['tools'] = tools
        if self.json_output:
            body['response_format'] = dict(type='json_object')
        request = urllib.request.Request(ENDPOINT, data=encode(body).encode(), method='POST',
            headers={'Authorization':'Bearer '+self.key, 'Content-Type':'application/json'})
        start = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'DeepSeek HTTP {exc.code}; no automatic retry') from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('DeepSeek network error; no automatic retry') from None
        self.last = dict(usage=normalize_usage(payload.get('usage',{}),'chat'),
                         api_latency_seconds=time.perf_counter()-start, response_model=payload.get('model'))
        choice = payload['choices'][0]
        self.last['finish_reason'] = choice.get('finish_reason')
        message = choice['message']
        if message.get('role') != 'assistant':
            raise RuntimeError('invalid model message role')
        clean = dict(role='assistant', content=message.get('content'))
        if message.get('tool_calls'):
            clean['tool_calls'] = message['tool_calls']
        return dict(self.last, message=clean)


def run_agent(messages, transport, plugin, tenant, session, mode='recover', max_steps=10,
              before_request=None, after_response=None, extra_tools=None, handlers=None,
              memory_tools=True, max_recovery_rounds=2, after_tools=None,
              snapshot_read_tools=None, write_guard=None, action_contract=None, eager_evidence=False):
    """Canonical state stays full; only each transport request is compacted."""
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    if max_recovery_rounds < 1:
        raise ValueError('positive recovery budget required')
    if plugin is None and (mode != 'full' or memory_tools):
        raise ValueError('native baseline requires full mode and memory_tools=False')
    history = copy.deepcopy(messages)
    steps, tool_seconds = [], 0.0
    context_controller = bool(plugin is not None and hasattr(plugin, 'context_tools'))
    if plugin is not None and not context_controller:
        raise ValueError("plugin must provide the current context tool interface")
    if eager_evidence and (plugin is None or not write_guard or not hasattr(write_guard,'evidence_views')):
        raise ValueError('eager evidence requires a context plugin and explicit fresh-read rules')
    extra_tools = write_guard.tools(extra_tools or []) if write_guard else copy.deepcopy(extra_tools or [])
    recovery_schemas = plugin.context_tools if context_controller else []
    tools = copy.deepcopy(recovery_schemas if memory_tools else [])+copy.deepcopy(extra_tools)
    handlers = handlers or {}
    names = [s['function']['name'] for s in tools]
    if len(names) != len(set(names)):
        raise ValueError('duplicate tool registration')
    snapshot_names = set(snapshot_read_tools or [])
    schemas = {s['function']['name']: s['function'] for s in extra_tools or []}
    actions = action_contract.start(schemas, handlers) if action_contract else None
    if write_guard and hasattr(write_guard, 'start'):
        write_guard = write_guard.start(schemas, handlers)
    if any(n not in schemas or schemas[n].get('parameters', {}).get('properties') for n in snapshot_names):
        raise ValueError('snapshot_read_tools must be explicitly registered zero-argument stable reads')
    read_snapshots = {}
    answer, outcome = '', 'step_limit'
    recovery_rounds, disabled_streak, context_streak = 0, 0, 0
    seen_context = set()
    full_locked, guard_reason = False, None
    for step in range(max_steps):
        start = time.perf_counter()
        if plugin is None:
            validate_messages(history)
            size = len(encode(history))
            prepared = PreparedHistory(copy.deepcopy(history), 'full', 'native_agent_without_plugin',
                size, size, 'characters', [], [], time.perf_counter()-start)
        else:
            if eager_evidence:
                prepared = plugin.before_model(tenant, session, history, 'full' if full_locked else mode,
                                               evidence_views=write_guard.evidence_views())
            else:
                prepared = plugin.before_model(tenant, session, history, 'full' if full_locked else mode)
        recovery_enabled = memory_tools and mode == 'recover' and (context_controller or prepared.mode == 'compressed')
        request_tools = copy.deepcopy(recovery_schemas if recovery_enabled else [])+copy.deepcopy(extra_tools)
        read_guard = memory_tools and mode == 'recover'
        suppressed_reads = sorted(n for n in read_snapshots
            if actions is None or not actions.needs_invocation(n, {})) if read_guard else []
        if write_guard or actions:
            # Request-only metadata lives at the current task boundary, never
            # before the old stable prefix or inside a tool-call group.
            offset = max(i for i,m in enumerate(prepared.messages) if m['role'] == 'user')+1
            notices = ([write_guard.notice()] if write_guard else []) + ([action_contract.notice()] if actions else [])
            for notice in reversed(notices):
                prepared.messages.insert(offset, dict(role='system', content=notice))
            prepared.sent_units = plugin._count(prepared.messages) if plugin else len(encode(prepared.messages))
            validate_messages(prepared.messages)
        log_start = time.perf_counter()
        if before_request:
            before_request(step, prepared)
        log_seconds = time.perf_counter()-log_start
        transport.last = {}
        response = None
        error_type = None
        try:
            response = transport(prepared.messages, request_tools)
            record = dict(response)
            message = response['message']
            calls = message.get('tool_calls', [])
            finish = response.get('finish_reason')
            status = 'truncated' if finish == 'length' else 'tool_calls' if calls else 'ok' if message.get('content') else 'empty'
            if status == 'ok' and actions and actions.missing():
                status = 'action_pending'
        except Exception as exc:
            record = dict(transport.last)
            error_type, status = type(exc).__name__, 'error'
        record.update(step=step, status=status, error_type=error_type,
                      history_mode=prepared.mode, history_reason=prepared.reason,
                      baseline_units=prepared.baseline_units, sent_units=prepared.sent_units,
                      unit=prepared.unit, omitted_turn_ids=prepared.omitted_turn_ids,
                      restored_turn_ids=prepared.restored_turn_ids,
                      evidence_plan=prepared.evidence_plan, recovery_tools_enabled=recovery_enabled,
                      context_metadata=prepared.context_metadata,
                      registered_tool_names=[t['function']['name'] for t in request_tools],
                      recovery_rounds=recovery_rounds, full_history_locked=full_locked,
                      recovery_guard_reason=guard_reason,
                      suppressed_snapshot_reads=suppressed_reads,
                      action_contract_state=actions.state() if actions else None,
                      preparation_seconds=prepared.preparation_seconds,
                      end_to_end_seconds=time.perf_counter()-start-log_seconds)
        steps.append(record)
        # Persist paid response usage before any tool can mutate external state.
        if after_response:
            after_response(record)
        if status in {'error', 'truncated', 'empty'}:
            outcome = status
            break
        history.append(message)
        if not calls:
            if actions and actions.missing():
                event = dict(error='required_actions_incomplete', executed=False, **actions.state())
                record['action_events'] = [event]
                history.append(dict(role='system', content=actions.retry_notice()))
                if after_tools:
                    after_tools(dict(step=step, memory_tool_events=[], action_events=[event],
                                     action_contract_state=actions.state(), action_receipts=[]))
                outcome = 'action_incomplete'
                continue
            answer, outcome = message['content'], 'ok'
            break
        tool_start = time.perf_counter()
        memory_events, recovery_called, disabled_only, blocked_read = [], False, True, False
        action_events, action_receipts = [], []
        write_event_start = len(write_guard.events) if write_guard and hasattr(write_guard, 'events') else 0
        for call in calls:
            invoked, handler_returned, checked = False, False, None
            try:
                function = call['function']
                arguments = json.loads(function['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('tool arguments must be an object')
                if context_controller and function['name'].startswith('context_'):
                    recovery_called, disabled_only = True, False
                    signature = encode([function['name'], arguments])
                    if not recovery_enabled:
                        value = dict(error='context_tools_disabled', progress=False)
                    elif signature in seen_context:
                        value = dict(error='duplicate_context_request', progress=False,
                                     note='Identical evidence already returned; use it or request a different source/page.')
                    elif recovery_rounds >= max_recovery_rounds:
                        value = dict(error='context_round_budget', progress=False,
                                     note='Finish with evidence already visible; no raw full-log fallback is performed.')
                    else:
                        value = plugin.invoke_context(tenant, session, function['name'], arguments)
                        seen_context.add(signature)
                    memory_events.append(dict(tool=function['name'], result=value))
                elif function['name'] in suppressed_reads and not arguments:
                    blocked_read = True
                    value = dict(status='already_read', previous_result=copy.deepcopy(read_snapshots[function['name']]),
                                 note='Use the identical prior result; this stable read was not executed again.')
                    memory_events.append(dict(tool=function['name'], result=value))
                elif function['name'] in handlers and function['name'] in {t['function']['name'] for t in request_tools}:
                    disabled_only = False
                    if actions:
                        actions.validate(function['name'], arguments)
                    if read_guard and function['name'] not in snapshot_names:
                        # Invalidate BEFORE invocation: a handler can partially
                        # mutate state and then raise. Its old snapshot is unsafe.
                        read_snapshots.clear()
                    checked = write_guard.validate(function['name'], arguments) if write_guard else arguments
                    if write_guard and hasattr(write_guard, 'before_invocation'):
                        write_guard.before_invocation(function['name'], checked)
                    invoked = True
                    value = handlers[function['name']](**checked)
                    handler_returned = True
                    if actions:
                        action_receipts.append(actions.record(function['name'], checked, value))
                    if read_guard:
                        if function['name'] in snapshot_names:
                            read_snapshots[function['name']] = copy.deepcopy(value)
                        else:
                            read_snapshots.clear()
                else:
                    disabled_only = False
                    raise ValueError('unregistered tool')
            except Exception as exc:
                disabled_only = False
                value = exc.result() if hasattr(exc, 'result') else dict(error=type(exc).__name__, note='tool failed; no facts recovered')
                if actions and invoked:
                    action_receipts.append(actions.record(function['name'], checked, error_type=type(exc).__name__))
                if actions and value.get('error') == 'action_prerequisite_missing':
                    action_events.append(dict(tool=call['function']['name'], result=value))
                if value.get('error') in {'provenance_rejected','write_value_rejected'}:
                    memory_events.append(dict(tool=call['function']['name'], result=value))
                if context_controller and call.get('function', {}).get('name', '').startswith('context_'):
                    recovery_called = True
                    memory_events.append(dict(tool=call['function']['name'], result=value))
            tool_message = dict(role='tool', tool_call_id=call['id'], content=encode(value))
            if invoked and write_guard and hasattr(write_guard, 'observe'):
                write_guard.observe(function['name'],checked,value,tool_message,len(history),handler_returned)
            history.append(tool_message)
        tool_seconds += time.perf_counter()-tool_start
        recovery_rounds += int(recovery_called)
        record['memory_tool_events'] = memory_events
        record['next_recovery_guard_reason'] = guard_reason
        record['action_events'] = action_events
        record['action_receipts'] = action_receipts
        record['action_contract_after'] = actions.state() if actions else None
        record['write_transform_events'] = (copy.deepcopy(write_guard.events[write_event_start:])
            if write_guard and hasattr(write_guard,'events') else [])
        if after_tools:
            after_tools(dict(step=step, memory_tool_events=memory_events,
                             next_recovery_guard_reason=guard_reason,
                             action_events=action_events, action_receipts=action_receipts,
                             action_contract_state=actions.state() if actions else None,
                             write_transform_events=record['write_transform_events']))
        disabled_streak = disabled_streak+1 if disabled_only else 0
        if memory_tools and mode == 'recover' and disabled_streak >= 2:
            outcome = 'tool_stalled' if blocked_read else 'recovery_stalled'
            break
        context_only = context_controller and all(c['function']['name'].startswith('context_') for c in calls)
        no_progress = context_only and memory_events and all(e['result'].get('error') or e['result'].get('results') == [] for e in memory_events)
        context_streak = context_streak+1 if no_progress else 0
        if context_streak >= 2:
            outcome = 'context_stalled'
            break
    if actions and actions.missing() and outcome in {'step_limit', 'action_incomplete'}:
        outcome = 'action_incomplete'
    elif actions and outcome == 'action_incomplete':
        # Obligations completed after a rejected final, but no new final arrived.
        outcome = 'step_limit'
    if plugin is not None:
        # Include final assistant/tool outputs even when no subsequent request
        # is made. Request-only controller notices never enter this archive.
        plugin.archive.ingest(scope_key(tenant, session), history)
    usages = [s['usage'] for s in steps if s.get('usage') is not None]
    return dict(answer=answer, status=outcome, messages=history, steps=steps,
                api_calls=len(steps), recovery_tool_calls=sum(c.get('function',{}).get('name','').startswith('context_')
                    for s in steps for c in s.get('message',{}).get('tool_calls',[])),
                prompt_tokens=sum(u['prompt_tokens'] for u in usages),
                completion_tokens=sum(u['completion_tokens'] for u in usages),
                unknown_usage_requests=sum(s.get('usage') is None for s in steps),
                action_contract=None if actions is None else dict(**actions.state(), receipts=actions.receipts),
                write_transform=write_guard.state() if write_guard and hasattr(write_guard,'state') else None,
                end_to_end_seconds=sum(s['end_to_end_seconds'] for s in steps)+tool_seconds,
                tool_seconds=tool_seconds)

