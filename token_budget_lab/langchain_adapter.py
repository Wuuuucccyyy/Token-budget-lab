"""LangChain middleware: compact requests, preserve state, verify actual tools."""
from __future__ import annotations

import asyncio
import copy
import inspect
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing_extensions import NotRequired

from .context_plugin import CONTEXT_TOOLS, ContextPlugin
from .langchain_model import canonical_messages, langchain_messages
from .memory import HistoryArchive, PreparedHistory, encode, scope_key, validate_messages
from .write_transform import FreshWriteGuard


@dataclass
class TokenBudgetContext:
    tenant: str
    session: str
    enabled: bool = True
    handlers: dict = field(default_factory=dict, repr=False)
    action_contract: Any = None
    write_guard: FreshWriteGuard | None = None
    write_arguments: dict = field(default_factory=dict)
    conditional_obligations: tuple[str, ...] = ()
    observer: Any = field(default=None, repr=False)


class TokenBudgetState(AgentState):
    token_budget_run: NotRequired[str]
    token_budget_result: NotRequired[dict]


def business_tools(schemas, handlers=None):
    """Wrap JSON-result business handlers; runtime handlers allow one shared graph."""
    result = []
    for schema in schemas:
        definition = schema['function']
        name = definition['name']
        if name.startswith(('context_', 'history_')):
            raise ValueError('Recovery tool names are reserved')

        def make(name):
            def call(runtime: ToolRuntime, **arguments):
                table = handlers if handlers is not None else runtime.context.handlers
                value = table[name](**arguments)
                if inspect.isawaitable(value):
                    raise TypeError('Use the async tool path for coroutine handlers')
                return encode(value), copy.deepcopy(value)

            async def acall(runtime: ToolRuntime, **arguments):
                table = handlers if handlers is not None else runtime.context.handlers
                function = table[name]
                if inspect.iscoroutinefunction(function):
                    value = await function(**arguments)
                else:
                    value = await asyncio.to_thread(function, **arguments)
                return encode(value), copy.deepcopy(value)
            return call, acall

        call, acall = make(name)
        result.append(StructuredTool.from_function(call, coroutine=acall, name=name,
            description=definition['description'], args_schema=copy.deepcopy(definition['parameters']),
            response_format='content_and_artifact'))
    return result


class _Run:
    def __init__(self, context, schemas, registered, options):
        self.context = context
        self.scope = scope_key(context.tenant, context.session)
        self.identity = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.async_tools = asyncio.Lock()
        self.started = time.perf_counter()
        self.actions = context.action_contract.start(schemas, registered) if context.action_contract else None
        self.guard = context.write_guard.start(schemas, registered) if context.write_guard else None
        self.write_arguments = {}
        for name, fields in context.write_arguments.items():
            if (name not in schemas or not isinstance(fields,(tuple,list)) or not fields or
                    any(not isinstance(f,str) or not f for f in fields) or len(set(fields))!=len(fields) or
                    not set(fields)<=set(schemas[name]['parameters'].get('properties',{}))):
                raise ValueError('Write argument scope must explicitly name registered tool fields')
            self.write_arguments[name] = frozenset(fields)
        self.conditional_obligations = frozenset(context.conditional_obligations)
        if self.conditional_obligations and (not self.guard or not self.guard.conditional_read_only or
                not self.actions or not self.conditional_obligations<=set(self.actions.contract.by_id)):
            raise ValueError('Conditional obligations require an explicit conditional source guard and contract')
        self.branch = self.guard.branch_state() if self.guard else None
        self.overlays, self.steps, self.tools = [], [], []
        self.seen_context, self.recovery_groups = set(), set()
        self.group_results = {}
        self.outcome = None
        self.worker, self.archive, self.plugin = None, None, None
        if context.enabled:
            # LangGraph executes model and tool nodes on different threads.
            # One owned worker keeps all SQLite/plugin operations on its thread.
            self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='context-archive')
            def initialize():
                self.archive = HistoryArchive(options['archive_dir'] / (self.scope + '.sqlite3'))
                self.plugin = ContextPlugin(self.archive, budget=options['budget'],
                    offload_tools=options['offload_tools'], offload_threshold=options['offload_threshold'])
            try:
                self.worker.submit(initialize).result()
            except BaseException:
                self.close()
                raise

    def archive_call(self, function):
        return self.worker.submit(function).result()

    def emit(self, event, **value):
        if self.context.observer:
            self.context.observer(dict(event=event, run_id=self.identity, **value))

    def waived(self):
        return self.conditional_obligations if self.branch and self.branch['selection_status'] in {'missing','ambiguous'} else frozenset()

    def missing(self):
        return [identity for identity in self.actions.missing() if identity not in self.waived()] if self.actions else []

    def action_state(self):
        if not self.actions:
            return None
        state=self.actions.state()
        state.update(complete=not self.missing(),missing=self.missing(),waived=sorted(self.waived()))
        return dict(**state,receipts=copy.deepcopy(self.actions.receipts))

    def close(self):
        if self.worker:
            if self.archive:
                self.worker.submit(self.archive.close).result()
            self.worker.shutdown(wait=True)
            self.worker = None


class TokenBudgetMiddleware(AgentMiddleware):
    """Toggle compression through invocation context on one compiled agent.

    Business tool executions are serialized within each invocation so reads,
    writes and receipts have a defined order. Independent sessions can run
    concurrently. Only model request views are replaced, never graph history.
    """
    state_schema = TokenBudgetState

    def __init__(self, tools, archive_dir, *, budget=9000, offload_threshold=1800,
                 offload_tools=('read_log',), max_steps=10, max_recovery_rounds=3,
                 eager_evidence=True):
        if type(max_steps) is not int or max_steps < 1 or type(max_recovery_rounds) is not int or max_recovery_rounds < 1:
            raise ValueError('Positive model and recovery limits required')
        self.business = {tool.name: tool for tool in tools}
        if len(self.business) != len(tools) or any(n.startswith(('context_', 'history_')) for n in self.business):
            raise ValueError('Unique business tools required')
        self.schemas = {schema['function']['name']: schema['function']
                        for schema in (convert_to_openai_tool(tool) for tool in tools)}
        self.options = dict(archive_dir=Path(archive_dir), budget=budget,
            offload_threshold=offload_threshold, offload_tools=tuple(offload_tools))
        self.max_steps, self.max_recovery_rounds = max_steps, max_recovery_rounds
        self.eager_evidence = eager_evidence
        self._runs, self._active = {}, {}
        self._lock = threading.RLock()
        self.tools = []
        for definition in CONTEXT_TOOLS:
            function = definition['function']
            def unavailable(**arguments):
                raise RuntimeError('Recovery requires TokenBudgetMiddleware interception')
            self.tools.append(StructuredTool.from_function(unavailable, name=function['name'],
                description=function['description'], args_schema=copy.deepcopy(function['parameters'])))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            runs = list(self._runs.values())
            self._runs.clear()
            self._active.clear()
        for run in runs:
            run.close()

    def _run(self, state):
        with self._lock:
            return self._runs[state['token_budget_run']]

    def _release(self, run):
        run.close()
        with self._lock:
            self._runs.pop(run.identity, None)
            self._active.pop(run.scope, None)

    def before_agent(self, state, runtime):
        context = runtime.context
        if not isinstance(context, TokenBudgetContext) or type(context.enabled) is not bool:
            raise ValueError('Explicit TokenBudgetContext required')
        if context.write_guard is not None and not isinstance(context.write_guard, FreshWriteGuard):
            raise ValueError('This adapter currently supports FreshWriteGuard')
        canonical_messages(state['messages'])
        scope = scope_key(context.tenant, context.session)
        with self._lock:
            if scope in self._active:
                raise ValueError('Concurrent invocations of the same tenant/session are unsupported')
            run = _Run(context, self.schemas, self.business, self.options)
            self._runs[run.identity] = run
            self._active[scope] = run.identity
        return dict(token_budget_run=run.identity, token_budget_result={})

    @hook_config(can_jump_to=['end'])
    def before_model(self, state, runtime):
        run = self._run(state)
        if run.outcome in {'context_stalled', 'error', 'truncated', 'empty'} or len(run.steps) >= self.max_steps:
            if run.outcome not in {'context_stalled', 'error', 'truncated', 'empty'}:
                run.outcome = 'action_incomplete' if run.missing() else 'step_limit'
            return dict(jump_to='end')
        return None

    def _prepare(self, request):
        run = self._run(request.state)
        canonical = canonical_messages(request.state['messages'])
        if canonical_messages(request.messages) != canonical:
            raise ValueError('Place TokenBudgetMiddleware before other message-rewriting middleware')
        validate_messages(canonical)
        start = time.perf_counter()
        if run.plugin:
            views = run.guard.evidence_views() if self.eager_evidence and run.guard else None
            prepared = run.archive_call(lambda: run.plugin.before_model(run.context.tenant,
                run.context.session, canonical, evidence_views=views))
        else:
            size = len(encode(canonical))
            prepared = PreparedHistory(copy.deepcopy(canonical), 'full', 'langchain_full_history',
                size, size, 'characters', [], [], time.perf_counter()-start)
        # The plugin preserves the complete current turn; translate canonical
        # positions to the prepared tail before inserting request-only notices.
        inserts = []
        for position, notice in run.overlays:
            index = len(prepared.messages) - (len(canonical) - position)
            if index < 0 or prepared.messages[index]['role'] != canonical[position]['role']:
                raise ValueError('Request-only notice anchor missing')
            inserts.append((index + 1, notice))
        offset = max(i for i, m in enumerate(prepared.messages) if m['role'] == 'user') + 1
        if run.actions:
            if run.conditional_obligations:
                notice=('[Current task execution contract] '+encode(run.context.action_contract.definition())+
                    '. Obligations '+encode(sorted(run.conditional_obligations))+
                    ' apply only when the actual fresh source is unique. They are waived only after a successful actual read establishes missing or ambiguous. '
                    'All other obligations remain required. A final answer is provisional until the applicable actions actually succeed. '
                    'Archive recovery and cached echoes are never current-run execution receipts.')
            else:
                notice=run.context.action_contract.notice()
            inserts.insert(0, (offset, notice))
        if run.write_arguments:
            inserts.append((offset,'[Current user write argument scope] '+encode({n:sorted(v) for n,v in run.write_arguments.items()})+
                '. Supply only these explicitly authorized fields. Optional schema fields are not automatically authorized. '
                'Source IDs and unrelated module fields are evidence metadata, not extra report values. Rejected calls do not write files; correct the arguments and finish the applicable task.'))
        if run.guard:
            inserts.insert(0, (offset, run.guard.notice()))
        for _, (index, notice) in sorted(enumerate(inserts), key=lambda x: (x[1][0], x[0]), reverse=True):
            prepared.messages.insert(index, dict(role='system', content=notice))
        validate_messages(prepared.messages)
        prepared.sent_units = len(encode(prepared.messages))
        prepared.context_metadata['conditional_branch'] = copy.deepcopy(run.branch)
        tools = request.tools if run.context.enabled else [t for t in request.tools
            if (t.name if hasattr(t, 'name') else t.get('function', {}).get('name')) not in
               {x['function']['name'] for x in CONTEXT_TOOLS}]
        step = dict(step=len(run.steps), history_mode=prepared.mode, history_reason=prepared.reason,
            baseline_units=prepared.baseline_units, sent_units=prepared.sent_units,
            context_metadata=copy.deepcopy(prepared.context_metadata),
            sent_messages=copy.deepcopy(prepared.messages), recovery_tools_enabled=run.context.enabled,
            preparation_seconds=prepared.preparation_seconds,
            registered_tool_names=[t.name if hasattr(t, 'name') else t['function']['name'] for t in tools])
        run.steps.append(step)
        run.emit('start', **step)
        return run, step, request.override(messages=langchain_messages(prepared.messages), tools=tools)

    def _response(self, run, step, response, elapsed):
        message = response.result[-1]
        step.update(message=canonical_messages([message])[0],
            usage=copy.deepcopy(message.response_metadata.get('token_budget_usage')),
            finish_reason=message.response_metadata.get('finish_reason'),
            model_seconds=elapsed)
        step['status'] = 'truncated' if step['finish_reason']=='length' else 'tool_calls' if message.tool_calls else 'ok' if message.content else 'empty'
        if step['usage'] is None and message.usage_metadata is not None:
            u = message.usage_metadata
            step['usage'] = dict(prompt_tokens=u['input_tokens'], completion_tokens=u['output_tokens'])
        run.emit('response', **{k:v for k,v in step.items() if k!='sent_messages'})

    def wrap_model_call(self, request, handler):
        run = self._run(request.state)
        try:
            run, step, modified = self._prepare(request)
        except BaseException:
            self._release(run)
            raise
        started = time.perf_counter()
        try:
            response = handler(modified)
        except Exception as exc:
            step.update(status='error', error_type=type(exc).__name__, usage=None)
            run.outcome = 'error'
            run.emit('response', **step)
            self._release(run)
            raise
        self._response(run, step, response, time.perf_counter()-started)
        return response

    async def awrap_model_call(self, request, handler):
        run = self._run(request.state)
        try:
            run, step, modified = await asyncio.to_thread(self._prepare, request)
        except BaseException:
            self._release(run)
            raise
        started = time.perf_counter()
        try:
            response = await handler(modified)
        except Exception as exc:
            step.update(status='error', error_type=type(exc).__name__, usage=None)
            run.outcome = 'error'
            run.emit('response', **step)
            self._release(run)
            raise
        self._response(run, step, response, time.perf_counter()-started)
        return response

    @hook_config(can_jump_to=['model', 'end'])
    def after_model(self, state, runtime):
        run = self._run(state)
        message = state['messages'][-1]
        if message.response_metadata.get('finish_reason') == 'length':
            run.outcome = 'truncated'
            return dict(jump_to='end')
        if not message.tool_calls:
            if not message.content:
                run.outcome = 'empty'
                return dict(jump_to='end')
            if run.missing():
                notice=('[Host finalization rejected] Missing applicable current-task execution receipts: '+
                    encode(run.missing())+'. The preceding answer is provisional. Execute the required authorized tools, then return the final answer. '
                    'Do not re-read old evidence instead of performing the actual operation; the original model-call cap still applies.')
                run.overlays.append((len(state['messages']) - 1, notice))
                run.outcome = 'action_incomplete'
                return dict(jump_to='model')
            run.outcome = 'ok'
        return None

    def _tool_start(self, request):
        run = self._run(request.state)
        call = request.tool_call
        name, arguments = call['name'], copy.deepcopy(call['args'])
        calls = request.state['messages'][-1].tool_calls
        ids = [c['id'] for c in calls]
        position = len(request.state['messages']) + ids.index(call['id'])
        group = tuple(ids)
        if name.startswith('context_'):
            signature = encode([name, arguments])
            if not run.context.enabled:
                value = dict(error='context_tools_disabled', progress=False)
            elif signature in run.seen_context:
                value = dict(error='duplicate_context_request', progress=False)
            elif group not in run.recovery_groups and len(run.recovery_groups) >= self.max_recovery_rounds:
                value = dict(error='context_round_budget', progress=False)
            else:
                run.recovery_groups.add(group)
                try:
                    value = run.archive_call(lambda: run.plugin.invoke_context(
                        run.context.tenant, run.context.session, name, arguments))
                    run.seen_context.add(signature)
                except Exception as exc:
                    value = dict(error=type(exc).__name__, progress=False)
            return run, position, group, arguments, ToolMessage(content=encode(value), tool_call_id=call['id'], artifact=value)
        if name not in self.business or request.tool is None:
            return run, position, group, arguments, ToolMessage(content=encode(dict(error='unregistered_tool')),
                tool_call_id=call['id'], status='error')
        try:
            if run.actions:
                run.actions.validate(name, arguments)
            if name in run.write_arguments and set(arguments)-run.write_arguments[name]:
                value=dict(error='write_arguments_rejected',executed=False,written=False,
                    unauthorized_fields=sorted(set(arguments)-run.write_arguments[name]),
                    allowed_fields=sorted(run.write_arguments[name]),
                    note='No write executed. Remove the unauthorized fields, submit the authorized write, then complete the required checks.')
                return run, position, group, arguments, ToolMessage(content=encode(value),tool_call_id=call['id'],artifact=value)
            if run.guard:
                arguments = run.guard.validate(name, arguments)
                run.guard.before_invocation(name, arguments)
        except Exception as exc:
            if not hasattr(exc, 'result'):
                raise
            value = exc.result()
            return run, position, group, arguments, ToolMessage(content=encode(value), tool_call_id=call['id'], artifact=value)
        return run, position, group, arguments, None

    def _tool_finish(self, request, prepared, message, invoked, error_type=None):
        run, position, group, arguments, _ = prepared
        name = request.tool_call['name']
        if not isinstance(message, ToolMessage):
            raise ValueError('This adapter requires ToolMessage results; Command tools are unsupported')
        value = message.artifact
        if value is None:
            import json
            try:
                value = json.loads(message.content)
            except (ValueError, TypeError):
                value = message.content
        receipt = None
        successful = invoked and error_type is None and message.status != 'error'
        if invoked:
            if run.actions:
                receipt = run.actions.record(name, arguments, value, error_type=error_type or
                    ('tool_error' if message.status == 'error' else None))
            if run.guard:
                run.guard.observe(name, arguments, value, canonical_messages([message])[0], position, successful)
                branch = run.guard.branch_state()
                if branch != run.branch:
                    run.branch = copy.deepcopy(branch)
                    run.overlays.append((len(request.state['messages']) + len(group) - 1,
                                         run.guard.branch_notice(branch)))
        record = dict(tool=name, arguments=copy.deepcopy(arguments), tool_call_id=message.tool_call_id,
            canonical_position=position, handler_invoked=invoked, handler_returned=successful,
            result=copy.deepcopy(value), receipt=receipt)
        run.tools.append(record)
        run.group_results.setdefault(group, []).append(record)
        if len(run.group_results[group]) == len(group) and all(c['name'].startswith('context_')
                for c in request.state['messages'][-1].tool_calls):
            groups = list(run.group_results.values())
            stalled = 0
            for rows in reversed(groups):
                if all(r['tool'].startswith('context_') and isinstance(r['result'], dict) and
                       (r['result'].get('error') or r['result'].get('results') == []) for r in rows):
                    stalled += 1
                else:
                    break
            if stalled >= 2:
                run.outcome = 'context_stalled'
        run.emit('tool', **record)
        return message

    def wrap_tool_call(self, request, handler):
        run = self._run(request.state)
        with run.lock:
            prepared = self._tool_start(request)
            message, invoked, error_type = prepared[-1], False, None
            if message is None:
                invoked = True
                try:
                    message = handler(request)
                except Exception as exc:
                    error_type = type(exc).__name__
                    value = dict(error=error_type, note='Actual business tool failed')
                    message = ToolMessage(content=encode(value), tool_call_id=request.tool_call['id'], artifact=value, status='error')
            return self._tool_finish(request, prepared, message, invoked, error_type)

    async def awrap_tool_call(self, request, handler):
        run = self._run(request.state)
        async with run.async_tools:
            prepared = await asyncio.to_thread(self._tool_start, request)
            message, invoked, error_type = prepared[-1], False, None
            if message is None:
                invoked = True
                try:
                    message = await handler(request)
                except Exception as exc:
                    error_type = type(exc).__name__
                    value = dict(error=error_type, note='Actual business tool failed')
                    message = ToolMessage(content=encode(value), tool_call_id=request.tool_call['id'], artifact=value, status='error')
            return self._tool_finish(request, prepared, message, invoked, error_type)

    def after_agent(self, state, runtime):
        run = self._run(state)
        canonical = canonical_messages(state['messages'])
        try:
            if run.archive:
                run.archive_call(lambda: run.archive.ingest(run.scope, canonical))
            usages = [s['usage'] for s in run.steps if s.get('usage') is not None]
            result = dict(status=run.outcome or 'step_limit',
                answer=canonical[-1]['content'] if run.outcome == 'ok' else '',
                enabled=run.context.enabled, api_calls=len(run.steps),
                prompt_tokens=sum(u['prompt_tokens'] for u in usages),
                completion_tokens=sum(u['completion_tokens'] for u in usages),
                unknown_usage_requests=sum(s.get('usage') is None for s in run.steps),
                recovery_tool_calls=sum(r['tool'].startswith('context_') for r in run.tools),
                end_to_end_seconds=time.perf_counter()-run.started,
                compressed_requests=sum(s['history_mode'] == 'compressed' for s in run.steps),
                action_contract=run.action_state(),
                write_argument_scope={name:sorted(fields) for name,fields in run.write_arguments.items()},
                write_transform=None if run.guard is None else run.guard.state())
            run.emit('complete', **result)
            return dict(token_budget_result=result)
        finally:
            self._release(run)
