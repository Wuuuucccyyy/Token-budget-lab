import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.action_contract import ActionContract, RequiredAction
from token_budget_lab.context_plugin import ContextPlugin, source_id
from token_budget_lab.memory import HistoryArchive, encode, validate_messages
from token_budget_lab.memory_agent import run_agent
from token_budget_lab.tool_tasks import tool
from helpers import Script


READ = tool('read_log', 'Read the current file.', dict(name=dict(type='string')), ['name'])
WRITE = tool('write_report', 'Write report.', dict(total=dict(type='string')), ['total'])
CHECK = tool('run_checks', 'Check current report.')
MESSAGES = [dict(role='system', content='Do the authorized local task.'),
            dict(role='user', content='Read scan_c now, write its result, then check.')]


def contract():
    return ActionContract([
        RequiredAction('fresh_read', 'read_log', dict(name='scan_c')),
        RequiredAction('write', 'write_report', after=('fresh_read',), result_matches=dict(written=True)),
        RequiredAction('check', 'run_checks', after=('write',), result_matches=dict(passed=True)),
    ], before_tools={'write_report': ['fresh_read'], 'run_checks': ['write']})


class ExecutionContractTests(unittest.TestCase):
    def run_script(self, responses, policy=None, handlers=None, messages=None, **kwargs):
        script = Script(responses)
        calls = []
        def handler(name, result):
            return lambda **args: calls.append((name, args)) or copy.deepcopy(result)
        tools = [READ, WRITE, CHECK]
        result = run_agent(messages or MESSAGES, script, None, 't', 's', mode='full', memory_tools=False,
            max_steps=len(responses), extra_tools=tools,
            handlers=handlers or {'read_log': handler('read_log', {'value': 204}),
                                  'write_report': handler('write_report', {'written': True}),
                                  'run_checks': handler('run_checks', {'passed': True})},
            action_contract=policy or contract(), **kwargs)
        validate_messages(result['messages'])
        return result, script, calls

    def test_finalization_is_checked_before_accepting_and_costs_all_attempts(self):
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'))])
        events = []
        result, script, calls = self.run_script(['done', [('read_log', {'name': 'scan_c'})], 'done'],
                                               policy=policy, after_tools=events.append)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(calls, [('read_log', {'name': 'scan_c'})])
        self.assertEqual(result['api_calls'], 3)
        self.assertEqual(result['prompt_tokens'], 300)
        self.assertEqual(result['completion_tokens'], 30)
        self.assertEqual(result['unknown_usage_requests'], 0)
        self.assertEqual(events[0]['action_events'][0]['error'], 'required_actions_incomplete')
        self.assertEqual(events[1]['action_receipts'][0]['matched_actions'], ['read'])
        self.assertTrue(any(m.get('content', '').startswith('[Host finalization rejected]')
                            for m in script.requests[1][0] if isinstance(m.get('content'), str)))

    def test_step_cap_is_not_extended_for_missing_reads(self):
        result, _, calls = self.run_script(['done'] * 3)
        self.assertEqual(result['status'], 'action_incomplete')
        self.assertEqual(result['answer'], '')
        self.assertEqual(result['api_calls'], 3)
        self.assertFalse(calls)
        self.assertEqual(result['action_contract']['missing'], ['fresh_read', 'write', 'check'])

    def test_no_side_effect_before_fresh_read_then_ordered_batch_can_finish(self):
        result, _, calls = self.run_script([
            [('write_report', {'total': '204.00'}), ('run_checks', {})],
            [('read_log', {'name': 'scan_c'}), ('write_report', {'total': '204.00'}), ('run_checks', {})],
            'done'])
        self.assertEqual([c[0] for c in calls], ['read_log', 'write_report', 'run_checks'])
        self.assertEqual(len(result['steps'][0]['action_events']), 2)
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['action_contract']['complete'])

    def test_wrong_file_is_an_actual_read_but_not_the_required_read(self):
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'))])
        result, _, calls = self.run_script([[('read_log', {'name': 'scan_a'})], 'done'], policy=policy)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['action_contract']['receipts'][0]['matched_actions'], [])
        self.assertEqual(result['status'], 'action_incomplete')

    def test_exception_and_returned_error_never_count_as_completion(self):
        invocations = []
        def read_log(name):
            invocations.append(name)
            if len(invocations) == 1:
                raise OSError('private details must not enter receipts')
            if len(invocations) == 2:
                return dict(error='not_read')
            return dict(value=204)
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'))])
        result, _, _ = self.run_script([[('read_log', {'name': 'scan_c'})]] * 3 + ['done'],
                                       policy=policy, handlers={'read_log': read_log})
        self.assertEqual(result['status'], 'ok')
        receipts = result['action_contract']['receipts']
        self.assertEqual([r['succeeded'] for r in receipts], [False, False, True])
        self.assertEqual([r['matched_actions'] for r in receipts], [[], [], ['read']])
        self.assertNotIn('private details', encode(receipts))

    def test_failed_checks_cannot_satisfy_successful_check_requirement(self):
        result, _, _ = self.run_script([
            [('read_log', {'name': 'scan_c'}), ('write_report', {'total': '204.00'}), ('run_checks', {})], 'done'],
            handlers={'read_log': lambda **args: {}, 'write_report': lambda **args: {'written': True},
                      'run_checks': lambda: {'passed': False}})
        self.assertEqual(result['action_contract']['missing'], ['check'])
        self.assertEqual(result['status'], 'action_incomplete')

    def test_later_write_invalidates_an_earlier_check(self):
        result, _, _ = self.run_script([
            [('read_log', {'name': 'scan_c'}), ('write_report', {'total': '204.00'}), ('run_checks', {})],
            [('write_report', {'total': '429.00'})], 'done'])
        self.assertEqual(result['action_contract']['missing'], ['check'])
        self.assertEqual(result['status'], 'action_incomplete')

    def test_history_and_context_read_do_not_satisfy_current_execution(self):
        old_result = dict(events=[dict(event='scan_c-0', status_code=204)], padding='x' * 5000)
        rows = MESSAGES[:1] + [dict(role='user', content='Read scan_c earlier.'),
            dict(role='assistant', content=None, tool_calls=[dict(id='old', type='function',
                 function=dict(name='read_log', arguments=encode(dict(name='scan_c'))))]),
            dict(role='tool', tool_call_id='old', content=encode(old_result)),
            dict(role='assistant', content='Read done.'), MESSAGES[-1]]
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'))])
        with tempfile.TemporaryDirectory() as tmp:
            archive = HistoryArchive(Path(tmp) / 'a.db')
            try:
                plugin = ContextPlugin(archive)
                script = Script([[('context_read', dict(source_id=source_id(3, rows[3]), pointer='/events/0/status_code'))], 'done'])
                actual = []
                result = run_agent(rows, script, plugin, 't', 's', max_steps=2,
                                   extra_tools=[READ], handlers={'read_log': lambda **args: actual.append(args)},
                                   action_contract=policy)
                self.assertEqual(result['status'], 'action_incomplete')
                self.assertEqual(result['action_contract']['receipts'], [])
                self.assertEqual(actual, [])
                self.assertEqual(result['recovery_tool_calls'], 1)
            finally:
                archive.close()

    def test_state_is_fresh_for_every_invocation(self):
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'))])
        first, _, _ = self.run_script([[('read_log', {'name': 'scan_c'})], 'done'], policy=policy)
        rows = first['messages'] + [dict(role='user', content='Read scan_c again now.')]
        second, _, calls = self.run_script(['done'], policy=policy, messages=rows)
        self.assertFalse(calls)
        self.assertEqual(second['status'], 'action_incomplete')
        self.assertEqual(second['action_contract']['receipts'], [])

    def test_static_contract_keeps_request_prefix_and_schemas_stable(self):
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'))])
        result, script, _ = self.run_script(['done', [('read_log', {'name': 'scan_c'})], 'done'], policy=policy)
        first = script.requests[0][0]
        for rows, schemas in script.requests[1:]:
            self.assertEqual(rows[:len(first)], first)
            self.assertEqual(schemas, script.requests[0][1])
        self.assertFalse(any(m.get('content') == policy.notice() for m in result['messages']))

    def test_cached_read_echo_does_not_create_another_receipt(self):
        read = tool('read_config', 'Stable current read.')
        policy = ActionContract([RequiredAction('read', 'read_config')])
        with tempfile.TemporaryDirectory() as tmp:
            archive = HistoryArchive(Path(tmp) / 'a.db')
            actual = []
            try:
                script = Script([[('read_config', {})], [('read_config', {})], 'done'])
                result = run_agent(MESSAGES, script, ContextPlugin(archive), 't', 's', max_steps=3,
                                   extra_tools=[read], handlers={'read_config': lambda: actual.append(True) or {}},
                                   snapshot_read_tools=['read_config'], action_contract=policy)
                self.assertEqual(actual, [True])
                self.assertEqual(len(result['action_contract']['receipts']), 1)
                self.assertEqual(result['status'], 'ok')
            finally:
                archive.close()

    def test_contract_rejects_bad_declarations_before_any_model_request(self):
        with self.assertRaises(ValueError):
            ActionContract([RequiredAction('x', 'context_read')])
        with self.assertRaises(ValueError):
            ActionContract([RequiredAction('a', 'read_log', after=('b',)), RequiredAction('b', 'read_log', after=('a',))])
        policy = ActionContract([RequiredAction('x', 'missing_handler')])
        script = Script(['done'])
        with self.assertRaises(ValueError):
            run_agent(MESSAGES, script, None, 't', 's', mode='full', memory_tools=False,
                      extra_tools=[READ], handlers={'read_log': lambda **args: {}}, action_contract=policy)
        self.assertEqual(script.requests, [])

    def test_typed_argument_and_nested_success_matching(self):
        policy = ActionContract([RequiredAction('read', 'read_log', dict(name='scan_c'),
                                                result_matches={'meta': {'fresh': True}})])
        tracker = policy.start(['read_log'], {'read_log': object()})
        tracker.record('read_log', {'name': 'scan_c'}, {'meta': {'fresh': 1}})
        self.assertEqual(tracker.missing(), ['read'])
        tracker.record('read_log', {'name': 'scan_c'}, {'meta': {'fresh': True}, 'other': 1})
        self.assertEqual(tracker.missing(), [])

    def test_two_ordered_identical_reads_require_two_actual_invocations(self):
        policy = ActionContract([
            RequiredAction('first', 'read_log', dict(name='scan_c')),
            RequiredAction('second', 'read_log', dict(name='scan_c'), after=('first',)),
        ])
        result, _, calls = self.run_script([[('read_log', {'name':'scan_c'})], 'done',
                                            [('read_log', {'name':'scan_c'})], 'done'], policy=policy)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(calls), 2)
        self.assertEqual([r['matched_actions'] for r in result['action_contract']['receipts']], [['first'], ['second']])

    def test_explicit_second_read_bypasses_stable_read_echo(self):
        read = tool('read_config', 'Current configuration.')
        policy = ActionContract([RequiredAction('first', 'read_config'),
                                 RequiredAction('second', 'read_config', after=('first',))])
        with tempfile.TemporaryDirectory() as tmp:
            archive = HistoryArchive(Path(tmp)/'a.db')
            actual = []
            try:
                script = Script([[('read_config', {})], [('read_config', {})], 'done'])
                result = run_agent(MESSAGES, script, ContextPlugin(archive), 't', 's', max_steps=3,
                    extra_tools=[read], handlers={'read_config': lambda: actual.append(True) or {}},
                    snapshot_read_tools=['read_config'], action_contract=policy)
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(len(actual), 2)
                self.assertEqual(len(result['action_contract']['receipts']), 2)
            finally:
                archive.close()


if __name__ == '__main__':
    unittest.main()
