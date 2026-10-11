"""Small Python agent integration with explicit, billable recovery steps."""
from __future__ import annotations
import copy
import json
import os
import time
import urllib.error
import urllib.request

from .deepseek import ENDPOINT, NoRedirect, normalize_usage
from .memory import HISTORY_TOOLS, PreparedHistory, encode, memory_tool, scope_key, validate_messages


FULL_HISTORY_NOTICE = (
    '[History controller status] The COMPLETE original conversation is included in this request. '
    'No archived turns are omitted. history_search, history_restore and history_restore_all are disabled. '
    'Read the visible original records directly, then use registered business tools to finish the task '
    'and return the requested final answer. Do not request history tools. '
    'This is availability metadata, not evidence about any task fact.')


def full_history_notice(prepared, plugin):
    """Add request-only status after original instructions, outside tool-call groups."""
    offset = next((i for i, m in enumerate(prepared.messages)
                   if m['role'] not in {'system', 'developer'}), len(prepared.messages))
    prepared.messages.insert(offset, dict(role='system', content=FULL_HISTORY_NOTICE))
    prepared.sent_units = plugin._count(prepared.messages)
    validate_messages(prepared.messages)


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


def run_agent(messages, transport, plugin, tenant, session, mode='recover', max_steps=4,
              before_request=None, after_response=None, extra_tools=None, handlers=None,
              memory_tools=True, max_recovery_rounds=2, reserve_task_steps=3, after_tools=None,
              snapshot_read_tools=None, write_guard=None):
    """Canonical state stays full; only each transport request is compacted."""
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    if max_recovery_rounds < 1 or reserve_task_steps < 1:
        raise ValueError('positive recovery and business-step budgets required')
    if plugin is None and (mode != 'full' or memory_tools):
        raise ValueError('native baseline requires full mode and memory_tools=False')
    history = copy.deepcopy(messages)
    steps, tool_seconds = [], 0.0
    context_controller = bool(plugin is not None and hasattr(plugin, 'context_tools'))
    extra_tools = write_guard.tools(extra_tools or []) if write_guard else copy.deepcopy(extra_tools or [])
    recovery_schemas = plugin.context_tools if context_controller else HISTORY_TOOLS
    tools = copy.deepcopy(recovery_schemas if memory_tools else [])+copy.deepcopy(extra_tools)
    handlers = handlers or {}
    names = [s['function']['name'] for s in tools]
    if len(names) != len(set(names)):
        raise ValueError('duplicate tool registration')
    snapshot_names = set(snapshot_read_tools or [])
    schemas = {s['function']['name']: s['function'] for s in extra_tools or []}
    if any(n not in schemas or schemas[n].get('parameters', {}).get('properties') for n in snapshot_names):
        raise ValueError('snapshot_read_tools must be explicitly registered zero-argument stable reads')
    read_snapshots = {}
    answer, outcome = '', 'step_limit'
    recovery_rounds, disabled_streak, context_streak = 0, 0, 0
    seen_searches, seen_context = set(), set()
    full_locked, saw_compression, guard_reason = False, False, None
    for step in range(max_steps):
        start = time.perf_counter()
        if plugin is None:
            validate_messages(history)
            size = len(encode(history))
            prepared = PreparedHistory(copy.deepcopy(history), 'full', 'native_agent_without_plugin',
                size, size, 'characters', [], [], time.perf_counter()-start)
        else:
            prepared = plugin.before_model(tenant, session, history, 'full' if full_locked else mode)
        if memory_tools and mode == 'recover' and not context_controller:
            if prepared.mode == 'compressed':
                saw_compression = True
                if recovery_rounds >= max_recovery_rounds or (recovery_rounds and max_steps-step <= reserve_task_steps):
                    guard_reason = 'recovery_round_budget' if recovery_rounds >= max_recovery_rounds else 'reserve_task_steps'
                    full_locked = True
                    prepared = plugin.before_model(tenant, session, history, 'full')
            elif saw_compression:
                full_locked = True
                guard_reason = guard_reason or 'full_history_fallback'
            if full_locked:
                prepared.mode, prepared.reason = 'fallback_full', guard_reason
                full_history_notice(prepared, plugin)
        # Full-history requests already contain all archived evidence. Register
        # recovery only on compressed requests, including after a manual restore.
        # The context tools also read small/older sources on a full shaped view.
        # Keep their schemas stable for a whole task, including after offload.
        recovery_enabled = memory_tools and mode == 'recover' and (context_controller or prepared.mode == 'compressed')
        request_tools = copy.deepcopy(recovery_schemas if recovery_enabled else [])+copy.deepcopy(extra_tools)
        read_guard = memory_tools and mode == 'recover'
        suppressed_reads = sorted(read_snapshots) if read_guard else []
        if not context_controller:
            request_tools = [t for t in request_tools if t['function']['name'] not in suppressed_reads]
        if suppressed_reads and not context_controller:
            offset = next((i for i, m in enumerate(prepared.messages)
                           if m['role'] not in {'system', 'developer'}), len(prepared.messages))
            prepared.messages.insert(offset, dict(role='system', content=
                '[Tool controller status] Stable read results already appear in the current tool chain. '
                + 'Temporarily disabled until another business action: '+', '.join(suppressed_reads)+'. '
                + f'There are {max_steps-step} model requests remaining, including this one. '
                + 'Use the prior results, finish required business actions and checks, then return the final answer. '
                + 'This status contains no task facts.'))
            prepared.sent_units = plugin._count(prepared.messages)
            validate_messages(prepared.messages)
        if write_guard:
            # Request-only metadata lives at the current task boundary, never
            # before the old stable prefix or inside a tool-call group.
            offset = max(i for i,m in enumerate(prepared.messages) if m['role'] == 'user')+1
            prepared.messages.insert(offset, dict(role='system', content=write_guard.notice()))
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
            answer, outcome = message['content'], 'ok'
            break
        tool_start = time.perf_counter()
        memory_events, recovery_called, disabled_only, blocked_read = [], False, True, False
        for call in calls:
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
                elif function['name'].startswith('history_'):
                    if not recovery_enabled:
                        value = dict(error='history_tools_disabled', progress=False,
                            full_history_available=prepared.mode != 'compressed',
                            note=('Complete original history is already visible. Read those records and finish with '
                                  'registered business tools or the final answer.' if prepared.mode != 'compressed' else
                                  'Recovery is disabled for this ablation.'))
                        if memory_tools and mode == 'recover':
                            full_locked, guard_reason = True, 'disabled_history_request'
                    else:
                        recovery_called, disabled_only = True, False
                        if function['name'] == 'history_search':
                            signature = encode(arguments)
                            if signature in seen_searches:
                                value = dict(progress=False, status='duplicate search; full history will be included next')
                                full_locked, guard_reason = True, 'duplicate_history_search'
                            else:
                                value = memory_tool(plugin, tenant, session, function['name'], arguments)
                                seen_searches.add(signature)
                        else:
                            value = memory_tool(plugin, tenant, session, function['name'], arguments)
                            if value.get('progress') is False:
                                full_locked, guard_reason = True, 'recovery_without_new_evidence'
                    memory_events.append(dict(tool=function['name'], result=value))
                elif function['name'] in suppressed_reads and not arguments:
                    blocked_read = True
                    value = dict(status='already_read', previous_result=copy.deepcopy(read_snapshots[function['name']]),
                                 note='Use the identical prior result; this stable read was not executed again.')
                    memory_events.append(dict(tool=function['name'], result=value))
                elif function['name'] in handlers and function['name'] in {t['function']['name'] for t in request_tools}:
                    disabled_only = False
                    if read_guard and function['name'] not in snapshot_names:
                        # Invalidate BEFORE invocation: a handler can partially
                        # mutate state and then raise. Its old snapshot is unsafe.
                        read_snapshots.clear()
                    checked = write_guard.validate(function['name'], arguments) if write_guard else arguments
                    value = handlers[function['name']](**checked)
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
                if value.get('error') == 'provenance_rejected':
                    memory_events.append(dict(tool=call['function']['name'], result=value))
                if call.get('function', {}).get('name', '').startswith('history_') and recovery_enabled:
                    recovery_called = True
                    full_locked, guard_reason = True, 'invalid_recovery_request'
                    memory_events.append(dict(tool=call['function']['name'], result=value))
                elif context_controller and call.get('function', {}).get('name', '').startswith('context_'):
                    recovery_called = True
                    memory_events.append(dict(tool=call['function']['name'], result=value))
            history.append(dict(role='tool', tool_call_id=call['id'], content=encode(value)))
        tool_seconds += time.perf_counter()-tool_start
        recovery_rounds += int(recovery_called)
        record['memory_tool_events'] = memory_events
        record['next_recovery_guard_reason'] = guard_reason
        if after_tools:
            after_tools(dict(step=step, memory_tool_events=memory_events,
                             next_recovery_guard_reason=guard_reason))
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
    if plugin is not None:
        # Include final assistant/tool outputs even when no subsequent request
        # is made. Request-only controller notices never enter this archive.
        plugin.archive.ingest(scope_key(tenant, session), history)
    usages = [s['usage'] for s in steps if s.get('usage') is not None]
    return dict(answer=answer, status=outcome, messages=history, steps=steps,
                api_calls=len(steps), recovery_tool_calls=sum(c.get('function',{}).get('name','').startswith(('history_','context_'))
                    for s in steps for c in s.get('message',{}).get('tool_calls',[])),
                prompt_tokens=sum(u['prompt_tokens'] for u in usages),
                completion_tokens=sum(u['completion_tokens'] for u in usages),
                unknown_usage_requests=sum(s.get('usage') is None for s in steps),
                end_to_end_seconds=sum(s['end_to_end_seconds'] for s in steps)+tool_seconds,
                tool_seconds=tool_seconds)
