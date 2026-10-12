"""Run the real LangChain graph with deterministic transports and actual tools."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

try:
    from langchain.agents import create_agent
except ImportError:
    create_agent = None

if create_agent is not None:
    from token_budget_lab import ActionContract, RequiredAction, FreshWriteGuard, WriteTransform
    from token_budget_lab.context_plugin import source_id
    from token_budget_lab.langchain_adapter import TokenBudgetContext, TokenBudgetMiddleware, business_tools
    from token_budget_lab.langchain_model import DeepSeekChatModel, canonical_messages, langchain_messages
    from token_budget_lab.memory import encode, validate_messages
    from token_budget_lab.tool_tasks import TOOLS


ROOT = Path(__file__).resolve().parents[1]


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.last = {}

    def __call__(self, messages, tools):
        self.requests.append((messages, tools))
        validate_messages(messages)
        response = next(self.responses)
        if callable(response):
            response = response(messages, tools)
        return dict(message=response, usage=dict(prompt_tokens=100, completion_tokens=5), finish_reason='stop')


def call(name, args, identity):
    return dict(role='assistant', content='', tool_calls=[dict(id=identity, type='function',
        function=dict(name=name, arguments=json.dumps(args)))])


@unittest.skipIf(create_agent is None, 'Install the langchain optional dependency')
class LangChainAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.invoked = []
        self.report = None
        self.payload = dict(events=[dict(event='target', status_code=204)], padding='x' * 6000)

        def read_log(name):
            self.invoked.append(('read_log', name))
            return self.payload

        def write_report(total, currency, recipient=None):
            self.invoked.append(('write_report', total))
            self.report = dict(total=total, currency=currency)
            if recipient is not None:self.report['recipient']=recipient
            return dict(written=True, report=self.report)

        def checks():
            self.invoked.append(('run_checks', None))
            return dict(passed=self.report is not None)

        self.handlers = {s['function']['name']: (lambda **args: {}) for s in TOOLS}
        self.handlers.update(read_log=read_log, write_report=write_report, run_checks=checks)
        self.tools = business_tools(TOOLS)
        self.guard = FreshWriteGuard([WriteTransform('read_log', dict(name='scan_c'), 'write_report', 'total',
            dict(kind='unique_event', array='/events', match_key='event', match_value='target', field='status_code'),
            fixed_arguments=dict(currency='USD'))])
        self.contract = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c')),
            RequiredAction('write', 'write_report', after=('read',), result_matches=dict(written=True)),
            RequiredAction('check', 'run_checks', after=('write',), result_matches=dict(passed=True))],
            before_tools=dict(write_report=['read'], run_checks=['write']))

    def graph(self, responses, **options):
        transport = ScriptedTransport(responses)
        middleware = TokenBudgetMiddleware(self.tools, Path(self.tmp.name)/'archive', **options)
        self.addCleanup(middleware.close)
        agent = create_agent(DeepSeekChatModel(transport=transport), tools=self.tools,
            middleware=[middleware], context_schema=TokenBudgetContext)
        return agent, middleware, transport

    def context(self, enabled=True, **overrides):
        values = dict(tenant='test', session='conversation', enabled=enabled,
            handlers=self.handlers, action_contract=self.contract, write_guard=self.guard)
        values.update(overrides)
        return TokenBudgetContext(**values)

    def sequence(self):
        return [call('read_log', dict(name='scan_c'), 'read'),
                call('write_report', dict(total='204.00', currency='USD'), 'write'),
                call('run_checks', {}, 'check'), dict(role='assistant', content='{"answer":"204.00"}')]

    def test_shared_graph_switch_preserves_actual_history_and_results(self):
        agent, middleware, transport = self.graph(self.sequence() + self.sequence())
        original = [dict(role='user', content='Read scan_c, write the target status with two decimals, then check.')]
        results = []
        for enabled in (False, True):
            self.report = None
            result = agent.invoke(dict(messages=langchain_messages(original)),
                context=self.context(enabled, session=str(enabled)), config=dict(max_concurrency=1, recursion_limit=50))
            results.append(result['token_budget_result'])
            actual = canonical_messages(result['messages'])
            read = next(m for m in actual if m['role']=='tool' and m['tool_call_id']=='read')
            self.assertEqual(json.loads(read['content']), self.payload)
            self.assertFalse(any(m['role']=='system' for m in actual))
            self.assertEqual(result['token_budget_result']['status'], 'ok')
            self.assertEqual(self.report, dict(total='204.00',currency='USD'))
        full, compact = transport.requests[1][0], transport.requests[5][0]
        self.assertIn('padding', full[-1]['content'])
        view=json.loads(compact[-1]['content'])
        self.assertTrue(view['archived'])
        self.assertEqual(view['requested_evidence'][0]['original_value'], 204)
        self.assertEqual(results[0]['answer'],results[1]['answer'])
        self.assertEqual(results[0]['api_calls'],4)
        self.assertEqual(results[1]['api_calls'],4)
        self.assertEqual(middleware._runs,{})

    def test_omitted_operation_rejects_early_final_without_auto_execution(self):
        responses=[dict(role='assistant',content='{"answer":"204.00"}'),*self.sequence()]
        agent,_,transport=self.graph(responses)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Read, write and check.')])),
            context=self.context(),config=dict(max_concurrency=1,recursion_limit=50))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(len(transport.requests),5)
        self.assertEqual(len(self.invoked),3)
        self.assertIn('Host finalization rejected',encode(transport.requests[1][0]))
        self.assertFalse(any('Host finalization rejected' in m['content'] for m in canonical_messages(result['messages'])))

    def test_wrong_numeric_scale_never_reaches_handler(self):
        sequence=self.sequence()
        sequence.insert(1,call('write_report',dict(total='2.04',currency='USD'),'wrong'))
        agent,_,_=self.graph(sequence)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Read and write the status.')])),
            context=self.context(),config=dict(max_concurrency=1,recursion_limit=50))
        self.assertEqual(result['token_budget_result']['write_transform']['rejected'],1)
        self.assertNotIn(('write_report','2.04'),self.invoked)
        self.assertEqual(self.report['total'],'204.00')

    def test_recovery_uses_exact_actual_source_on_langgraph_tool_thread(self):
        def recovery(messages, tools):
            view=json.loads(messages[-1]['content'])
            return call('context_find',dict(source_id=view['source_id'],match_key='event',
                match_value='target',fields=['status_code']),'recover')
        sequence=self.sequence();sequence.insert(1,recovery)
        agent,_,transport=self.graph(sequence,eager_evidence=False)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Read and recover target.')])),
            context=self.context(),config=dict(max_concurrency=1,recursion_limit=50))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(result['token_budget_result']['recovery_tool_calls'],1)
        evidence=json.loads(transport.requests[2][0][-1]['content'])
        self.assertEqual(evidence['matches'][0]['fields']['status_code'],204)
        raw=canonical_messages(result['messages'])
        read=next((i,m) for i,m in enumerate(raw) if m.get('tool_call_id')=='read')
        self.assertEqual(result['token_budget_result']['write_transform']['events'][0]['source_id'],source_id(*read))

    def test_async_agent_executes_and_preserves_archive(self):
        agent,_,_=self.graph(self.sequence())
        result=asyncio.run(agent.ainvoke(dict(messages=langchain_messages([dict(role='user',content='Read and write.')])),
            context=self.context(),config=dict(max_concurrency=1,recursion_limit=50)))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(result['token_budget_result']['action_contract']['completed'],['read','write','check'])

    def test_conditional_read_only_feedback_and_attempt_receipts(self):
        self.payload=dict(events=[],padding='x'*6000)
        guard=FreshWriteGuard(self.guard.rules,conditional_read_only=True)
        responses=[call('read_log',dict(name='scan_c'),'read'),
            call('write_report',dict(total='UNKNOWN',currency='USD'),'forbidden'),
            dict(role='assistant',content='{"answer":"UNKNOWN"}')]
        agent,_,transport=self.graph(responses)
        context=self.context(action_contract=ActionContract([RequiredAction('read','read_log',dict(name='scan_c'))]),write_guard=guard)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='If missing, return UNKNOWN without writes.')])),
            context=context,config=dict(max_concurrency=1,recursion_limit=50))
        self.assertIsNone(self.report)
        self.assertEqual(len(self.invoked),1)
        self.assertIn('conditional_action_rejected',transport.requests[2][0][-1]['content'])
        self.assertIn('read-only branch',encode(transport.requests[1][0]))

    def test_model_call_cap_is_not_a_graph_recursion_guess(self):
        agent,_,transport=self.graph([call('read_log',dict(name='scan_c'),f'read-{i}') for i in range(3)],max_steps=3)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Read.') ])),
            context=self.context(action_contract=None,write_guard=None),config=dict(recursion_limit=50))
        self.assertEqual(result['token_budget_result']['status'],'step_limit')
        self.assertEqual(len(transport.requests),3)
        self.assertEqual(result['token_budget_result']['answer'],'')

    def test_full_disables_recovery_schemas(self):
        agent,_,transport=self.graph([dict(role='assistant',content='done')])
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Answer.') ])),
            context=self.context(False,action_contract=None,write_guard=None))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertFalse(any(t['function']['name'].startswith('context_') for t in transport.requests[0][1]))

    def test_complete_multi_tool_group_has_matching_source_positions(self):
        sequence=self.sequence()
        sequence[0]['tool_calls'].extend(call('read_report',{},'other')['tool_calls'])
        agent,_,_=self.graph(sequence)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Read and write.')])),
            context=self.context(),config=dict(max_concurrency=2,recursion_limit=50))
        raw=canonical_messages(result['messages'])
        validate_messages(raw)
        read=next((i,m) for i,m in enumerate(raw) if m.get('tool_call_id')=='read')
        self.assertEqual(result['token_budget_result']['write_transform']['events'][0]['source_id'],source_id(*read))
        self.assertEqual(result['token_budget_result']['status'],'ok')

    def test_request_prefix_and_tool_schemas_stay_stable(self):
        agent,_,transport=self.graph(self.sequence())
        agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Read and write.')])),
            context=self.context(),config=dict(max_concurrency=1,recursion_limit=50))
        for previous,current in zip(transport.requests,transport.requests[1:]):
            self.assertEqual(previous[0],current[0][:len(previous[0])])
            self.assertEqual(previous[1],current[1])

    def test_network_error_is_not_retried_and_releases_invocation(self):
        def fail(messages, tools):
            raise RuntimeError('simulated transport failure')
        agent,middleware,transport=self.graph([fail,dict(role='assistant',content='done')])
        context=self.context(False,action_contract=None,write_guard=None)
        messages=dict(messages=langchain_messages([dict(role='user',content='Answer.')] ))
        with self.assertRaises(RuntimeError):agent.invoke(messages,context=context)
        self.assertEqual(len(transport.requests),1)
        self.assertFalse(middleware._runs)
        result=agent.invoke(messages,context=context)
        self.assertEqual(result['token_budget_result']['status'],'ok')

    def test_unsupported_multimodal_message_is_rejected_before_dispatch(self):
        from langchain_core.messages import HumanMessage
        agent,_,transport=self.graph([])
        with self.assertRaises(ValueError):
            agent.invoke(dict(messages=[HumanMessage(content=[dict(type='image_url',image_url=dict(url='file://example'))])]),
                context=self.context())
        self.assertEqual(len(transport.requests),0)

    def test_unauthorized_optional_field_is_blocked_and_correction_is_metered(self):
        sequence=self.sequence()
        sequence.insert(1,call('write_report',dict(total='204.00',currency='USD',recipient='beta@example.invalid'),'extra'))
        sequence.insert(2,dict(role='assistant',content='{"answer":"204.00"}'))
        agent,_,transport=self.graph(sequence)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Write only total and currency, then check.')])),
            context=self.context(write_arguments={'write_report':['total','currency']}),config=dict(max_concurrency=1,recursion_limit=50))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(len(transport.requests),6)
        self.assertIn('write_arguments_rejected',transport.requests[2][0][-1]['content'])
        self.assertEqual(self.report,dict(total='204.00',currency='USD'))
        self.assertEqual(sum(name=='write_report' for name,_ in self.invoked),1)

    def test_explicitly_authorized_optional_field_is_allowed(self):
        sequence=self.sequence()
        sequence[1]['tool_calls'][0]['function']['arguments']=encode(dict(total='204.00',currency='USD',recipient='allowed@example.invalid'))
        agent,_,_=self.graph(sequence)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='Include the authorized recipient.')])),
            context=self.context(write_arguments={'write_report':['total','currency','recipient']}),config=dict(max_concurrency=1,recursion_limit=50))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(self.report['recipient'],'allowed@example.invalid')

    def test_conditional_missing_waives_writes_but_does_not_mark_them_executed(self):
        self.payload=dict(events=[],padding='x'*6000)
        guard=FreshWriteGuard(self.guard.rules,conditional_read_only=True)
        sequence=[call('read_log',dict(name='scan_c'),'read'),dict(role='assistant',content='{"answer":"UNKNOWN"}')]
        agent,_,_=self.graph(sequence)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='If missing do not write; otherwise write and check.')])),
            context=self.context(write_guard=guard,conditional_obligations=('write','check')),
            config=dict(max_concurrency=1,recursion_limit=50))
        actions=result['token_budget_result']['action_contract']
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(actions['completed'],['read'])
        self.assertEqual(actions['waived'],['check','write'])
        self.assertEqual(len(actions['receipts']),1)
        self.assertIsNone(self.report)

    def test_conditional_unique_requires_actual_write_and_check(self):
        sequence=self.sequence()
        sequence.insert(1,dict(role='assistant',content='{"answer":"204.00"}'))
        agent,_,transport=self.graph(sequence)
        result=agent.invoke(dict(messages=langchain_messages([dict(role='user',content='If unique write and check.')])),
            context=self.context(write_guard=FreshWriteGuard(self.guard.rules,conditional_read_only=True),conditional_obligations=('write','check')),
            config=dict(max_concurrency=1,recursion_limit=50))
        self.assertEqual(result['token_budget_result']['status'],'ok')
        self.assertEqual(result['token_budget_result']['action_contract']['waived'],[])
        self.assertEqual(result['token_budget_result']['action_contract']['completed'],['read','write','check'])
        self.assertEqual(len(transport.requests),5)


if __name__=='__main__':unittest.main()
