"""Small Python agent integration with explicit, billable recovery steps."""
from __future__ import annotations
import copy
import json
import os
import time
import urllib.error
import urllib.request

from .deepseek import ENDPOINT, NoRedirect, normalize_usage
from .memory import HISTORY_TOOLS, PreparedHistory, encode, memory_tool, validate_messages


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
              memory_tools=True):
    """Canonical state stays full; only each transport request is compacted."""
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    if plugin is None and (mode != 'full' or memory_tools):
        raise ValueError('native baseline requires full mode and memory_tools=False')
    history = copy.deepcopy(messages)
    steps, tool_seconds = [], 0.0
    tools = copy.deepcopy(HISTORY_TOOLS if memory_tools else [])+copy.deepcopy(extra_tools or [])
    handlers = handlers or {}
    names = [s['function']['name'] for s in tools]
    if len(names) != len(set(names)):
        raise ValueError('duplicate tool registration')
    answer, outcome = '', 'step_limit'
    for step in range(max_steps):
        start = time.perf_counter()
        if plugin is None:
            validate_messages(history)
            size = len(encode(history))
            prepared = PreparedHistory(copy.deepcopy(history), 'full', 'native_agent_without_plugin',
                size, size, 'characters', [], [], time.perf_counter()-start)
        else:
            prepared = plugin.before_model(tenant, session, history, mode)
        # Full-history requests already contain all archived evidence. Register
        # recovery only on compressed requests, including after a manual restore.
        recovery_enabled = memory_tools and prepared.mode == 'compressed' and mode == 'recover'
        request_tools = copy.deepcopy(HISTORY_TOOLS if recovery_enabled else [])+copy.deepcopy(extra_tools or [])
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
                      registered_tool_names=[t['function']['name'] for t in request_tools],
                      preparation_seconds=prepared.preparation_seconds,
                      end_to_end_seconds=time.perf_counter()-start-log_seconds)
        steps.append(record)
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
        for call in calls:
            try:
                function = call['function']
                arguments = json.loads(function['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('tool arguments must be an object')
                if function['name'].startswith('history_'):
                    if not recovery_enabled:
                        raise ValueError('history tools are not registered')
                    if mode == 'compact':
                        value = dict(error='recovery disabled for this ablation; do not guess missing facts')
                    else:
                        value = memory_tool(plugin, tenant, session, function['name'], arguments)
                elif function['name'] in handlers:
                    value = handlers[function['name']](**arguments)
                else:
                    raise ValueError('unregistered tool')
            except Exception as exc:
                value = dict(error=type(exc).__name__, note='tool failed; no facts recovered')
            history.append(dict(role='tool', tool_call_id=call['id'], content=encode(value)))
        tool_seconds += time.perf_counter()-tool_start
    usages = [s['usage'] for s in steps if s.get('usage') is not None]
    return dict(answer=answer, status=outcome, messages=history, steps=steps,
                api_calls=len(steps), recovery_tool_calls=sum(c.get('function',{}).get('name','').startswith('history_')
                    for s in steps for c in s.get('message',{}).get('tool_calls',[])),
                prompt_tokens=sum(u['prompt_tokens'] for u in usages),
                completion_tokens=sum(u['completion_tokens'] for u in usages),
                unknown_usage_requests=sum(s.get('usage') is None for s in steps),
                end_to_end_seconds=sum(s['end_to_end_seconds'] for s in steps)+tool_seconds,
                tool_seconds=tool_seconds)
