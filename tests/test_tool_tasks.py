import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from token_budget_lab.experiment import ROOT
from token_budget_lab.longmemeval_adapter import convert
from token_budget_lab.memory import validate_messages
from token_budget_lab.memory_agent import DeepSeekChat,run_agent
from token_budget_lab.scripted_policy import ScriptedPolicy
from token_budget_lab.tool_experiment import execute
from token_budget_lab.tool_tasks import TOOLS, ToolSandbox, assess, build_tasks, money


class ToolTaskTests(unittest.TestCase):
    def test_fixture_reproduction_and_episode_split(self):
        data=json.loads((ROOT/'data/agent_tool_tasks.json').read_text(encoding='utf-8'))
        self.assertEqual(data,build_tasks())
        self.assertEqual(len(data['episodes']),24)
        self.assertEqual(sum(len(e['checkpoints']) for e in data['episodes']),72)
        dev={e['id'] for e in data['episodes'] if e['split']=='dev'}
        test={e['id'] for e in data['episodes'] if e['split']=='test'}
        self.assertFalse(dev&test)
        self.assertEqual(len(dev),12)
        self.assertEqual(len(test),12)

    def test_actual_files_csv_checks_and_path_restriction(self):
        e=build_tasks()['episodes'][0]
        with tempfile.TemporaryDirectory() as directory:
            sandbox=ToolSandbox(Path(directory)/'episode',e)
            self.assertEqual(sandbox.read_config()['config'],e['initial_config'])
            total=sandbox.invoke('calculate_total',dict(items=[dict(quantity=2,price='1.25'),dict(quantity=3,price='2.50')]))
            self.assertEqual(total['total'],'10.00')
            sandbox.invoke('write_report',dict(total='10.00',currency='USD'))
            self.assertEqual(json.loads((sandbox.path/'report.json').read_text())['total'],'10.00')
            result=sandbox.invoke('run_checks',{})
            self.assertTrue(result['passed'])
            self.assertIn('Ran 2 tests',result['stderr'])
            original=sandbox.file_hashes()
            with self.assertRaises(ValueError):sandbox.invoke('read_log',dict(name='../config'))
            self.assertEqual(original,sandbox.file_hashes())
            with self.assertRaises(ValueError):sandbox.invoke('write_config',dict(timeout_ms=True))
            with self.assertRaises(ValueError):money([dict(quantity=1,price='NaN')])

    def test_answer_alone_does_not_pass_action_task(self):
        e=build_tasks()['episodes'][0]
        with tempfile.TemporaryDirectory() as directory:
            sandbox=ToolSandbox(Path(directory)/'episode',e)
            sandbox.seed_history(e)
            result=dict(status='ok',answer=json.dumps(dict(answer=e['checkpoints'][0]['answer'])))
            score=assess(e['checkpoints'][0],result,sandbox,sandbox.file_hashes(),0,e['initial_config'])
            self.assertTrue(score['answer_correct'])
            self.assertTrue(score['state_correct'])
            self.assertFalse(score['required_tools_completed'])
            self.assertFalse(score['task_success'])

    def test_unknown_with_an_unauthorized_write_fails_even_if_answer_matches(self):
        e=next(e for e in build_tasks()['episodes'] if e['kind']=='abstention')
        with tempfile.TemporaryDirectory() as directory:
            sandbox=ToolSandbox(Path(directory)/'episode',e)
            sandbox.seed_history(e)
            before=sandbox.file_hashes()
            sandbox.invoke('write_report',dict(recipient='beta@example.invalid'))
            result=dict(status='ok',answer='{"answer":"UNKNOWN"}')
            score=assess(e['checkpoints'][0],result,sandbox,before,0,e['initial_config'])
            self.assertTrue(score['answer_correct'])
            self.assertTrue(score['constraint_violation'])
            self.assertFalse(score['task_success'])

    def test_reference_changes_do_not_change_seed_messages(self):
        e=build_tasks()['episodes'][0]
        altered=copy.deepcopy(e)
        for c in altered['checkpoints']:c.update(answer='SECRET_REFERENCE',timeout_ms=123456)
        with tempfile.TemporaryDirectory() as directory:
            original,_=ToolSandbox(Path(directory)/'a',e).seed_history(e)
            other,_=ToolSandbox(Path(directory)/'b',altered).seed_history(altered)
        self.assertEqual(original,other)
        self.assertNotIn('SECRET_REFERENCE',str(other))
        validate_messages(original)

    def test_continuous_native_driver_writes_historical_values_without_memory_tools(self):
        e=next(e for e in build_tasks()['episodes'] if e['kind']=='temporal')
        with tempfile.TemporaryDirectory() as directory:
            sandbox=ToolSandbox(Path(directory)/'episode',e)
            history,_=sandbox.seed_history(e)
            original_length=len(history)
            client=ScriptedPolicy()
            for checkpoint in e['checkpoints']:
                history.append(dict(role='user',content=checkpoint['query']))
                start=len(sandbox.trace);before=sandbox.file_hashes()
                result=run_agent(history,client,None,'t','s',mode='full',memory_tools=False,
                    extra_tools=TOOLS,handlers=sandbox.handlers(),max_steps=6)
                self.assertTrue(assess(checkpoint,result,sandbox,before,start,e['initial_config'])['task_success'])
                self.assertEqual(result['steps'][0]['history_reason'],'native_agent_without_plugin')
                self.assertEqual(result['recovery_tool_calls'],0)
                history=result['messages']
            self.assertGreater(len(history),original_length)
            for checkpoint in e['checkpoints']:
                self.assertIn(dict(role='user',content=checkpoint['query']),history)
            validate_messages(history)

    def test_offline_summary_does_not_claim_llm_accuracy(self):
        config=json.loads((ROOT/'configs/agent_tools_offline.json').read_text())
        config.update(per_kind_limit=1,modes=['full','recover'])
        with tempfile.TemporaryDirectory() as directory:
            summary=execute(config,Path(directory)/'run')
        self.assertEqual(summary['api_attempts'],0)
        for run in summary['runs']:
            self.assertIsNone(run['task_success_rate'])
            self.assertIsNone(run['prompt_tokens'])
            self.assertEqual(run['diagnostic_driver_success'],12)
            self.assertEqual(run['quality_status'],'scripted_driver_only_not_llm_accuracy')

    def test_mid_task_paid_limit_keeps_usage_and_actual_tool_audit(self):
        config=json.loads((ROOT/'configs/agent_tools_deepseek.json').read_text())
        config['max_calls']=1
        class Transport:
            last={}
            def __call__(self,messages,tools):
                return dict(message=dict(role='assistant',content=None,tool_calls=[dict(id='action',type='function',
                    function=dict(name='write_config',arguments='{"timeout_ms":321}'))]),
                    finish_reason='tool_calls',usage=dict(prompt_tokens=17,completion_tokens=9))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run'
            with self.assertRaises(RuntimeError):execute(config,output,Transport())
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['api_attempts'],1)
            self.assertEqual(sum(r['prompt_tokens'] for r in summary['runs']),17)
            self.assertEqual(sum(r['completion_tokens'] for r in summary['runs']),9)
            self.assertEqual(sum(r['checkpoints'] for r in summary['runs']),0)
            records=[json.loads(l) for l in (output/'requests.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['event']=='business_tool' and r['tool']=='write_config' for r in records))
            with self.assertRaises(ValueError):execute(config,output,Transport())


class OfficialAdapterTests(unittest.TestCase):
    def row(self,qid='q1',kind='knowledge-update'):
        return dict(question_id=qid,question_type=kind,question='What is the current value?',answer=99,
            question_date='2026-10-11',haystack_session_ids=['later','earlier'],haystack_dates=['2026-10-05','2026-10-01'],
            haystack_sessions=[[dict(role='user',content='current=99',has_answer=True)],
                              [dict(role='user',content='old=42',has_answer=False)]])

    def test_dates_are_sorted_and_labels_never_enter_messages(self):
        row=self.row();row['answer']='SECRET_REFERENCE'
        bundle=convert([row],variant='s')
        s=bundle['sessions'][0]
        self.assertLess(str(s['messages']).index('old=42'),str(s['messages']).index('current=99'))
        self.assertNotIn('has_answer',str(s['messages']))
        self.assertNotIn('SECRET_REFERENCE',str(s['messages']))
        self.assertTrue(s['questions'][0]['evidence_positions'])
        self.assertFalse(bundle['oracle_warning'])

    def test_abstention_keeps_original_reference_and_flags_oracle(self):
        row=self.row(qid='q_abs')
        bundle=convert([row],variant='oracle')
        q=bundle['sessions'][0]['questions'][0]
        self.assertEqual(q['kind'],'abstention')
        self.assertEqual(q['answers'],['UNKNOWN'])
        self.assertEqual(q['official_reference_answer'],'99')
        self.assertTrue(bundle['oracle_warning'])
        with self.assertRaises(ValueError):convert([row,row])


class JsonTransportTests(unittest.TestCase):
    def test_json_mode_and_native_request_omit_empty_tool_schemas(self):
        import io
        class Opener:
            def open(self,request,timeout):
                body=json.loads(request.data)
                self_test.assertNotIn('tools',body)
                self_test.assertEqual(body['response_format'],{'type':'json_object'})
                return io.BytesIO(json.dumps(dict(model='test',usage=dict(prompt_tokens=8,completion_tokens=3),
                    choices=[dict(finish_reason='stop',message=dict(role='assistant',content='{"answer":"ok"}'))])).encode())
        self_test=self
        with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-key'}):
            client=DeepSeekChat(model='test',json_output=True)
            client.opener=Opener()
            result=client([dict(role='user',content='output json')],[])
        self.assertEqual(result['usage']['prompt_tokens'],8)
