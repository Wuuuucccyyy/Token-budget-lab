import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.memory import HistoryArchive, MemoryPlugin, memory_tool, scope_key, validate_messages
from token_budget_lab.memory_agent import run_agent
from token_budget_lab.memory_experiment import execute, prepare


def history():
    rows = [dict(role='system',content='Use records. Never infer unavailable facts.'),
            dict(role='user',content='Do not publish. Project constraint: output JSON only.')]
    for n in range(4):
        rows += [dict(role='user',content=f'Inspect record {n}.'),
                 dict(role='assistant',content=None,tool_calls=[dict(id=f'c{n}',type='function',
                      function=dict(name='read_record',arguments=json.dumps(dict(n=n))))]),
                 dict(role='tool',tool_call_id=f'c{n}',content=f'EXACT record {n}: retry_count={n+3}. '+('background '*200)),
                 dict(role='assistant',content=f'Inspection {n} completed.')]
    rows.append(dict(role='user',content='Return record 0 retry_count.'))
    return rows


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = HistoryArchive(Path(self.temp.name)/'history.sqlite3')
        self.plugin = MemoryPlugin(self.archive, budget=4000, recent_turns=1, retrieval=False)

    def tearDown(self):
        self.archive.close()
        self.temp.cleanup()

    def test_canonical_state_is_unchanged_and_user_constraints_remain(self):
        rows = history()
        original = copy.deepcopy(rows)
        result = self.plugin.before_model('tenant','session',rows)
        self.assertEqual(rows, original)
        self.assertEqual(self.archive.read(scope_key('tenant','session')),original)
        self.assertEqual(result.mode,'compressed')
        for row in rows:
            if row['role'] in {'user','system'}:
                self.assertIn(row,result.messages)
        self.assertLess(result.sent_units,result.baseline_units)
        validate_messages(result.messages)

    def test_restore_in_original_roles_and_order(self):
        rows=history()
        first=self.plugin.before_model('tenant','session',rows)
        scope=scope_key('tenant','session')
        matches=self.plugin.search(scope,'record 0 retry_count')
        self.assertTrue(matches)
        self.plugin.restore(scope,['T2'])
        second=self.plugin.before_model('tenant','session',rows)
        restored=rows[2:6]
        for message in restored:
            self.assertIn(message,second.messages)
        positions=[second.messages.index(m) for m in restored]
        self.assertEqual(positions,sorted(positions))
        self.assertIn('T2',second.restored_turn_ids)
        self.assertTrue(first.omitted_turn_ids)

    def test_complete_current_tool_chain_is_preserved(self):
        rows=history()
        rows += [dict(role='assistant',content=None,tool_calls=[dict(id='latest',type='function',
                         function=dict(name='read_record',arguments='{}'))]),
                 dict(role='tool',tool_call_id='latest',content='current result')]
        result=self.plugin.before_model('tenant','session',rows)
        self.assertEqual(result.messages[-3:],rows[-3:])
        validate_messages(result.messages)

    def test_pending_or_orphan_calls_rejected(self):
        rows=history()[:-2]
        with self.assertRaises(ValueError):
            self.plugin.before_model('tenant','session',[dict(role='tool',tool_call_id='missing',content='x')])
        with self.assertRaises(ValueError):
            validate_messages(history()[:4])

    def test_tenant_and_session_isolation(self):
        self.plugin.before_model('tenant','one',history())
        self.assertEqual(self.archive.read(scope_key('tenant','two')),[])
        self.assertEqual(self.archive.read(scope_key('other','one')),[])
        self.assertEqual(self.plugin.search(scope_key('other','one'),'retry_count'),[])
        with self.assertRaises(ValueError):
            memory_tool(self.plugin,'tenant','two','history_restore',{'turn_ids':['T2']})

    def test_history_edits_shrinks_and_corruption_rejected(self):
        rows=history()
        self.plugin.before_model('tenant','session',rows)
        altered=copy.deepcopy(rows)
        altered[0]['content']='different system'
        with self.assertRaises(ValueError):
            self.plugin.before_model('tenant','session',altered)
        with self.assertRaises(ValueError):
            self.plugin.before_model('tenant','session',rows[:-1])
        self.archive.connection.execute('UPDATE history SET value=? WHERE position=0',('{"role":"user"}',))
        with self.assertRaises(ValueError):
            self.archive.read(scope_key('tenant','session'))

    def test_budget_is_soft_when_quality_protection_needs_full_history(self):
        plugin=MemoryPlugin(self.archive,budget=100,recent_turns=1)
        rows=history()
        result=plugin.before_model('tenant','session',rows)
        self.assertEqual(result.mode,'fallback_full')
        self.assertEqual(result.messages,rows)

    def test_agent_can_request_entire_history(self):
        rows=history()
        self.plugin.before_model('tenant','session',rows)
        memory_tool(self.plugin,'tenant','session','history_restore_all',{})
        result=self.plugin.before_model('tenant','session',rows)
        self.assertEqual(result.messages,rows)
        self.assertEqual(result.reason,'agent_requested_full_history')

    def test_restore_requests_expire_on_new_user_turn(self):
        rows=history()
        self.plugin.before_model('tenant','session',rows)
        self.plugin.restore(scope_key('tenant','session'),full=True)
        rows += [dict(role='assistant',content='3'),dict(role='user',content='next question')]
        result=self.plugin.before_model('tenant','session',rows)
        self.assertNotEqual(result.reason,'agent_requested_full_history')

    def test_identifier_lookup_recovers_source_despite_format_words(self):
        rows=history()
        rows[-1]['content']='EXACT record 0 retry_count 是多少？只返回数字字符串。'
        # Exercise a source with a unique identifier instead of repeated labels.
        rows[2]['content']='Read ALPHA snapshot.'
        rows[4]['content']='ALPHA retry_count=3. '+('background '*200)
        rows[-1]['content']='ALPHA 的 retry_count 是多少？只返回数字字符串。'
        plugin=MemoryPlugin(self.archive,budget=5000,recent_turns=1)
        result=plugin.before_model('tenant','auto',rows)
        self.assertEqual(result.mode,'compressed')
        self.assertIn(rows[4],result.messages)
        self.assertIn(rows[-1],result.messages)

    def test_unknown_and_multimodal_content_fall_back(self):
        plugin=MemoryPlugin(self.archive,budget=4000,recent_turns=1)
        rows=history()
        rows[-1]['content']='unrecorded billing_contact_name'
        result=plugin.before_model('tenant','unknown',rows)
        self.assertEqual(result.mode,'fallback_full')
        rows=history()
        rows[-1]['content']=[dict(type='text',text='question')]
        result=plugin.before_model('tenant','multimodal',rows)
        self.assertEqual(result.messages,rows)
        self.assertEqual(result.reason,'unsupported_content_preserved')


class AgentIntegrationTests(unittest.TestCase):
    def test_host_tools_execute_and_keep_the_canonical_chain(self):
        class Transport:
            last={}
            calls=0
            def __call__(self,messages,tools):
                self.calls+=1
                if self.calls==1:
                    self_test.assertIn('add',[t['function']['name'] for t in tools])
                    message=dict(role='assistant',content=None,tool_calls=[dict(id='host1',type='function',
                        function=dict(name='add',arguments='{"a":2,"b":3}'))])
                else:
                    self_test.assertEqual(json.loads(messages[-1]['content']),5)
                    message=dict(role='assistant',content='5')
                return dict(message=message,usage=dict(prompt_tokens=20,completion_tokens=4),finish_reason='stop')
        self_test=self
        tool=dict(type='function',function=dict(name='add',description='add integers',
            parameters=dict(type='object',properties=dict(a=dict(type='integer'),b=dict(type='integer')),required=['a','b'])))
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'memory.sqlite3')
            try:
                result=run_agent([dict(role='user',content='add two and three')],Transport(),MemoryPlugin(archive),
                    't','s',extra_tools=[tool],handlers={'add':lambda a,b:a+b})
            finally:
                archive.close()
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['answer'],'5')
        self.assertEqual(result['recovery_tool_calls'],0)
        validate_messages(result['messages'])

    def test_truncated_answer_is_not_returned_as_success_and_is_metered(self):
        class Transport:
            last={}
            def __call__(self,messages,tools):
                return dict(message=dict(role='assistant',content='partial'),
                    usage=dict(prompt_tokens=10,completion_tokens=8),finish_reason='length')
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'memory.sqlite3')
            try:
                result=run_agent([dict(role='user',content='question')],Transport(),MemoryPlugin(archive),'t','s')
            finally:
                archive.close()
        self.assertEqual(result['status'],'truncated')
        self.assertEqual(result['answer'],'')
        self.assertEqual(result['completion_tokens'],8)

    def test_recovery_loop_billing_and_hooks(self):
        class Transport:
            last={}
            calls=0
            def __call__(self,messages,tools):
                self.calls+=1
                if self.calls==1:
                    message=dict(role='assistant',content=None,tool_calls=[dict(id='restore1',type='function',
                        function=dict(name='history_restore',arguments='{"turn_ids":["T2"]}'))])
                else:
                    self_test.assertTrue(any(m.get('tool_call_id')=='c0' for m in messages))
                    message=dict(role='assistant',content='3')
                return dict(message=message,usage=dict(prompt_tokens=100,completion_tokens=10),finish_reason='stop')
        self_test=self
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'memory.sqlite3')
            plugin=MemoryPlugin(archive,budget=4000,recent_turns=1,retrieval=False)
            starts,ends=[],[]
            result=run_agent(history(),Transport(),plugin,'tenant','session',
                before_request=lambda n,p:starts.append(n),after_response=ends.append)
            archive.close()
        self.assertEqual(result['answer'],'3')
        self.assertEqual(result['api_calls'],2)
        self.assertEqual(result['prompt_tokens'],200)
        self.assertEqual(result['completion_tokens'],20)
        self.assertEqual(result['recovery_tool_calls'],1)
        self.assertEqual(starts,[0,1])
        self.assertEqual(len(ends),2)

    def test_known_usage_survives_failed_paid_response(self):
        class Transport:
            last={}
            def __call__(self,messages,tools):
                self.last=dict(usage=dict(prompt_tokens=17,completion_tokens=9))
                raise RuntimeError('private provider body')
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'memory.sqlite3')
            result=run_agent(history(),Transport(),MemoryPlugin(archive),'t','s')
            archive.close()
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['prompt_tokens'],17)
        self.assertEqual(result['api_calls'],1)
        self.assertNotIn('private provider body',str(result))

    def test_experiment_counts_complete_tasks_and_limits_attempts(self):
        from token_budget_lab.experiment import ROOT
        config=json.loads((ROOT/'configs/agent_memory_deepseek.json').read_text(encoding='utf-8'))
        config.update(session_limit=1,max_calls=1)
        class Transport:
            last={}
            calls=0
            def __call__(self,messages,tools):
                self.calls+=1
                return dict(message=dict(role='assistant',content='{"answer":"UNKNOWN"}'),
                            usage=dict(prompt_tokens=10,completion_tokens=5),finish_reason='stop')
        client=Transport()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                execute(config,Path(directory),client)
            summary=json.loads((Path(directory)/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['api_attempts'],1)
            self.assertEqual(sum(r['prompt_tokens'] for r in summary['runs']),10)
            self.assertEqual(client.calls,1)
            with self.assertRaises(ValueError):
                execute(config,Path(directory),client)

    def test_data_sessions_are_separate_and_ground_truth_not_in_planning(self):
        from token_budget_lab.experiment import ROOT
        from scripts.prepare_agent_memory import build
        bundle=json.loads((ROOT/'data/agent_memory.json').read_text(encoding='utf-8'))
        self.assertEqual(bundle,build())
        splits={s:{r['id'] for r in bundle['sessions'] if r['split']==s} for s in ('dev','test')}
        self.assertFalse(splits['dev']&splits['test'])
        self.assertEqual(len(bundle['sessions']),20)
        self.assertEqual(sum(len(s['questions']) for s in bundle['sessions']),120)

    def test_mid_task_call_limit_preserves_usage_without_a_final_task(self):
        from token_budget_lab.experiment import ROOT
        config=json.loads((ROOT/'configs/agent_memory_deepseek.json').read_text(encoding='utf-8'))
        config.update(session_limit=1,max_calls=1)
        class Transport:
            last={}
            def __call__(self,messages,tools):
                return dict(message=dict(role='assistant',content=None,tool_calls=[dict(id='mem',type='function',
                    function=dict(name='history_restore_all',arguments='{}'))]),
                    usage=dict(prompt_tokens=15,completion_tokens=6),finish_reason='tool_calls')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                execute(config,Path(directory),Transport())
            summary=json.loads((Path(directory)/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(sum(r['prompt_tokens'] for r in summary['runs']),15)
            self.assertEqual(sum(r['completion_tokens'] for r in summary['runs']),6)
            self.assertEqual(sum(r['questions'] for r in summary['runs']),0)


class FormatDiagnosticTests(unittest.TestCase):
    def test_post_hoc_parser_does_not_pick_numbers_from_explanations(self):
        from scripts.analyze_agent_memory import read_literal
        self.assertEqual(read_literal('203'),('203','bare_scalar'))
        self.assertEqual(read_literal('false'),('false','bare_scalar'))
        self.assertEqual(read_literal('{"answer":"323"}'),('323','strict_json'))
        self.assertEqual(read_literal('125 + 132 = 257'),(None,'needs_review'))
        self.assertEqual(read_literal('total 323\n{"answer":"323"}'),(None,'needs_review'))


if __name__=='__main__':
    unittest.main()
