import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.memory import HistoryArchive, MemoryPlugin, scope_key, validate_messages
from token_budget_lab.memory_agent import FULL_HISTORY_NOTICE, run_agent


def records():
    rows = [dict(role='system', content='Use original records; output JSON.')]
    for n in range(4):
        rows += [dict(role='user', content=f'Inspect record {n}'),
                 dict(role='assistant', content=f'record {n}: value={n+3}. '+('background '*200))]
    return rows+[dict(role='user', content='Use record 0 and finish the file task.')]


def call(name, arguments):
    return name, arguments


class Script:
    last = {}
    def __init__(self, responses):
        self.responses, self.requests = responses, []

    def __call__(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        response = self.responses[len(self.requests)-1]
        message = dict(role='assistant', content=response if isinstance(response, str) else None)
        if not isinstance(response, str):
            message['tool_calls'] = [dict(id=f'{len(self.requests)}-{i}', type='function',
                function=dict(name=name, arguments=json.dumps(args))) for i, (name, args) in enumerate(response)]
        return dict(message=message, usage=dict(prompt_tokens=100, completion_tokens=10))


BUSINESS = dict(type='function', function=dict(name='finish', description='Complete the file task',
                parameters=dict(type='object', properties={}, additionalProperties=False)))


class RecoveryGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = HistoryArchive(Path(self.temp.name)/'memory.sqlite3')
        self.plugin = MemoryPlugin(self.archive, budget=4000, recent_turns=1, retrieval=False)
        self.scope = scope_key('t', 's')

    def tearDown(self):
        self.archive.close()
        self.temp.cleanup()

    def run_script(self, responses, **kwargs):
        script = Script(responses)
        actions = []
        result = run_agent(records(), script, self.plugin, 't', 's', max_steps=6,
            extra_tools=[BUSINESS], handlers={'finish': lambda: actions.append('finished') or {'done': True}}, **kwargs)
        validate_messages(result['messages'])
        return result, script, actions

    def test_restore_is_idempotent_and_search_labels_visible_records(self):
        self.plugin.before_model('t', 's', records())
        first = self.plugin.restore(self.scope, ['T1'])
        repeated = self.plugin.restore(self.scope, ['T1'])
        self.assertTrue(first['progress'])
        self.assertFalse(repeated['progress'])
        self.assertEqual(repeated['already_requested_turn_ids'], ['T1'])
        self.plugin.before_model('t', 's', records())
        visible = self.plugin.restore(self.scope, ['T1'])
        self.assertFalse(visible['progress'])
        self.assertEqual(visible['already_visible_turn_ids'], ['T1'])
        matches = self.plugin.search(self.scope, 'record 0 value')
        self.assertTrue(next(r for r in matches if r['turn_id'] == 'T1')['already_visible'])

    def test_duplicate_restore_falls_back_once_then_business_completes(self):
        result, script, actions = self.run_script([
            [call('history_restore', {'turn_ids': ['T1']})],
            [call('history_restore', {'turn_ids': ['T1']})],
            [call('finish', {})], '{"answer":"done"}'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(actions, ['finished'])
        self.assertEqual([s['history_mode'] for s in result['steps']],
                         ['compressed', 'compressed', 'fallback_full', 'fallback_full'])
        self.assertEqual(result['steps'][2]['history_reason'], 'recovery_without_new_evidence')
        self.assertEqual(result['prompt_tokens'], 400)
        self.assertTrue(any(m.get('content') == FULL_HISTORY_NOTICE for m in script.requests[2][0]))
        self.assertFalse(any(m.get('content') == FULL_HISTORY_NOTICE for m in result['messages']))
        self.assertTrue(all(m in script.requests[2][0] for m in records()))

    def test_duplicate_search_escalates_even_before_round_cap(self):
        result, _, _ = self.run_script([
            [call('history_search', {'query': 'record 0 value'})],
            [call('history_search', {'query': 'record 0 value'})], 'done'], max_recovery_rounds=5)
        self.assertEqual(result['steps'][2]['history_reason'], 'duplicate_history_search')
        self.assertEqual(result['recovery_tool_calls'], 2)

    def test_round_budget_preserves_business_steps_without_increasing_max_steps(self):
        self.plugin.budget = 6000
        result, script, actions = self.run_script([
            [call('history_search', {'query': 'record 0'})],
            [call('history_restore', {'turn_ids': ['T1']})],
            [call('finish', {})], 'done'])
        self.assertEqual(result['steps'][2]['history_reason'], 'recovery_round_budget')
        self.assertEqual(actions, ['finished'])
        self.assertEqual([t['function']['name'] for t in script.requests[2][1]], ['finish'])

    def test_disabled_tool_feedback_names_available_history_and_stops_stubborn_loop(self):
        result, _, _ = self.run_script([
            [call('history_restore_all', {})],
            [call('history_search', {'query': 'record 0'})],
            [call('history_search', {'query': 'record 0'})]])
        self.assertEqual(result['status'], 'recovery_stalled')
        self.assertEqual(result['api_calls'], 3)
        self.assertEqual(result['prompt_tokens'], 300)
        self.assertEqual(result['answer'], '')
        payload = json.loads(result['messages'][-1]['content'])
        self.assertTrue(payload['full_history_available'])
        self.assertEqual(payload['error'], 'history_tools_disabled')

    def test_invalid_recovery_and_parallel_calls_keep_protocol_and_business_actions(self):
        result, _, actions = self.run_script([
            [call('history_restore', {'turn_ids': ['missing']}), call('finish', {})], 'done'])
        self.assertEqual(result['steps'][1]['history_reason'], 'invalid_recovery_request')
        self.assertEqual(actions, ['finished'])
        self.assertEqual(result['status'], 'ok')

    def test_query_scope_reset_does_not_leave_full_history_locked(self):
        result, _, _ = self.run_script([[call('history_restore_all', {})], 'done'])
        new = result['messages']+[dict(role='user', content='Use record 1.')]
        script = Script(['done'])
        again = run_agent(new, script, self.plugin, 't', 's', max_steps=6)
        self.assertEqual(again['steps'][0]['history_mode'], 'compressed')
        self.assertFalse(again['steps'][0]['full_history_locked'])

    def test_native_baseline_receives_no_controller_notice(self):
        script = Script(['done'])
        result = run_agent(records(), script, None, 't', 's', mode='full', memory_tools=False, max_steps=6)
        self.assertEqual(script.requests[0][0], records())
        self.assertEqual(result['status'], 'ok')

    def test_response_accounting_precedes_business_actions_and_guard_event_follows(self):
        order = []
        script = Script([[call('finish', {})], 'done'])
        run_agent(records(), script, self.plugin, 't', 's', max_steps=6, extra_tools=[BUSINESS],
                  handlers={'finish': lambda: order.append('action')},
                  after_response=lambda r: order.append('paid_response'),
                  after_tools=lambda r: order.append('guard_event'))
        self.assertEqual(order, ['paid_response', 'action', 'guard_event', 'paid_response'])

    def test_stable_read_is_suppressed_and_invalidated_after_write(self):
        read_tool = copy.deepcopy(BUSINESS)
        read_tool['function']['name'] = 'read_state'
        script = Script([[call('read_state', {})], [call('read_state', {})],
                         [call('finish', {})], [call('read_state', {})], 'done'])
        state, reads = {'value': 3}, []
        def read():
            reads.append(state['value'])
            return dict(state)
        def write():
            state['value'] = 9
            return dict(state)
        result = run_agent(records(), script, self.plugin, 't', 's', max_steps=6,
            extra_tools=[read_tool, BUSINESS], handlers={'read_state': read, 'finish': write},
            snapshot_read_tools=['read_state'])
        self.assertEqual(reads, [3, 9])
        self.assertNotIn('read_state', [t['function']['name'] for t in script.requests[1][1]])
        self.assertIn('read_state', [t['function']['name'] for t in script.requests[3][1]])
        self.assertEqual(result['prompt_tokens'], 500)
        validate_messages(result['messages'])

    def test_stubborn_stable_read_loop_stops_and_records_every_paid_call(self):
        read_tool = copy.deepcopy(BUSINESS)
        read_tool['function']['name'] = 'read_state'
        reads = []
        result = run_agent(records(), Script([[call('read_state', {})]]*3), self.plugin,
            't', 's', max_steps=6, extra_tools=[read_tool],
            handlers={'read_state': lambda: reads.append(1) or {'value': 3}}, snapshot_read_tools=['read_state'])
        self.assertEqual(result['status'], 'tool_stalled')
        self.assertEqual(reads, [1])
        self.assertEqual(result['api_calls'], 3)
        self.assertEqual(result['prompt_tokens'], 300)

    def test_stable_read_guard_rejects_undeclared_and_parameterized_tools(self):
        with self.assertRaises(ValueError):
            run_agent(records(), Script(['done']), self.plugin, 't', 's', snapshot_read_tools=['unknown'])
        parameterized = copy.deepcopy(BUSINESS)
        parameterized['function']['parameters']['properties'] = {'path': {'type': 'string'}}
        with self.assertRaises(ValueError):
            run_agent(records(), Script(['done']), self.plugin, 't', 's',
                      extra_tools=[parameterized], snapshot_read_tools=['finish'])

    def test_remaining_steps_reserve_applies_before_recovery_round_cap(self):
        self.plugin.budget = 6000
        result, _, _ = self.run_script([[call('history_search', {'query': 'record 0'})],
                                        [call('finish', {})], [call('finish', {})], 'done'],
                                       max_recovery_rounds=5)
        self.assertEqual(result['steps'][3]['history_reason'], 'reserve_task_steps')

    def test_search_excludes_active_tool_chain_and_previews_source_content(self):
        rows = records()
        self.plugin.before_model('t', 's', rows)
        self.assertNotIn('T9', [r['turn_id'] for r in self.plugin.search(self.scope, 'finish file')])


if __name__ == '__main__':
    unittest.main()
