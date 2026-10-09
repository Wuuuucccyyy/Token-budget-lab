import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from token_budget_lab.core import CounterBackend, compress
from token_budget_lab.datasets import load_rows
from token_budget_lab.deepseek import DeepSeek
from token_budget_lab.experiment import ROOT, execute, plan
from token_budget_lab.metrics import score_answer
from token_budget_lab.analysis import cluster_ci


class ExperimentTests(unittest.TestCase):
    def test_neighbor_budget_order_and_radius_zero_ablation(self):
        text = 'Background details. Mercury is a planet. It has no moons. Other planets have moons.'
        counter=CounterBackend()
        for budget in range(len(text)+1):
            result=compress('Mercury',text,budget,counter,'bm25_neighbor')
            self.assertLessEqual(len(result),budget)
            positions=[text.index(s) for s in result.splitlines()]
            self.assertEqual(positions,sorted(positions))
            self.assertEqual(compress('Mercury',text,budget,counter,'bm25'),
                             compress('Mercury',text,budget,counter,'bm25_neighbor',radius=0))
        self.assertIn('It has no moons.',compress('Mercury',text,65,counter,'bm25_neighbor'))

    def test_alternative_references_and_token_overlap(self):
        self.assertEqual(score_answer('The Denver Broncos.', ['Broncos', 'Denver Broncos'])['exact_match'],1)
        self.assertAlmostEqual(score_answer('Denver Broncos team',['Denver Broncos'])['f1'],.8)
        self.assertEqual(score_answer('', ['Denver'])['f1'],0)

    def test_public_splits_are_disjoint_and_references_valid(self):
        splits={s:load_rows(ROOT/'data/squad_article.json',s) for s in ['train','dev','test']}
        docs={s:{r['document_id'] for r in rows} for s,rows in splits.items()}
        self.assertEqual([len(splits[s]) for s in ['train','dev','test']],[280,100,100])
        self.assertFalse(docs['train'] & docs['test'] or docs['dev'] & docs['test'] or docs['train'] & docs['dev'])

    def config(self):
        return dict(backend='deepseek', data='data/toy_qa.jsonl',limit=2,seed=1,
                    model='fake',protocol='chat',thinking='disabled',max_tokens=100,max_calls=2,
                    strategies=[dict(name='full',method='full',cache='exact')])

    def fake(self, empty=False):
        class Fake:
            calls=0
            last={}
            def __call__(self,q,c):
                self.calls+=1
                self.last=dict(usage=dict(prompt_tokens=50,completion_tokens=10),finish_reason='stop')
                return '' if empty else '不开放'
        return Fake()

    def test_resume_preserves_results_and_does_not_repeat_api_calls(self):
        config=self.config()
        client=self.fake()
        with tempfile.TemporaryDirectory() as directory:
            first=execute(config,Path(directory),client)
            second=execute(config,Path(directory),client,resume=True)
            self.assertEqual(client.calls,1)
            self.assertEqual(first['runs'],second['runs'])
            self.assertEqual(second['runs'][0]['cache_hits'],1)
            config['max_tokens']=200
            with self.assertRaisesRegex(ValueError,'changed'):
                execute(config,Path(directory),client,resume=True)

    def test_interrupted_offline_work_is_resumed(self):
        config=dict(backend='offline',data='data/toy_qa.jsonl',limit=3,
                    strategies=[dict(name='full',method='full')])
        with tempfile.TemporaryDirectory() as directory:
            with patch('token_budget_lab.experiment.extractive_answer',side_effect=['first',RuntimeError('interrupt')]):
                with self.assertRaises(RuntimeError):
                    execute(config,Path(directory))
            with patch('token_budget_lab.experiment.extractive_answer',return_value='rest') as answer:
                result=execute(config,Path(directory),resume=True)
            self.assertEqual(answer.call_count,2)
            self.assertEqual(result['runs'][0]['requests'],3)

    def test_empty_answers_are_logged_but_never_cached(self):
        client=self.fake(empty=True)
        with tempfile.TemporaryDirectory() as directory:
            result=execute(self.config(),Path(directory),client)
            self.assertEqual(client.calls,2)
            self.assertEqual(result['runs'][0]['empty'],2)
            self.assertEqual(result['runs'][0]['prompt_tokens'],100)

    def test_unknown_paid_failure_is_not_silently_retried(self):
        client=self.fake()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(type(client),'__call__',side_effect=RuntimeError('private server text')):
                with self.assertRaisesRegex(RuntimeError,'preserved'):
                    execute(self.config(),Path(directory),client)
            summary=json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(summary['runs'][0]['unknown_usage_requests'],1)
            with self.assertRaisesRegex(RuntimeError,'no automatic'):
                execute(self.config(),Path(directory),client,resume=True)

    def test_pending_request_blocks_replay(self):
        client=self.fake()
        with tempfile.TemporaryDirectory() as directory:
            execute(self.config(),Path(directory),client)
            with (Path(directory)/'requests.jsonl').open('a') as log:
                log.write(json.dumps(dict(event='start',key='unresolved'))+'\n')
            with self.assertRaisesRegex(RuntimeError,'unresolved'):
                execute(self.config(),Path(directory),client,resume=True)

    def test_call_limit_is_enforced(self):
        config=self.config()
        config['strategies'][0]['cache']='none'
        config['max_calls']=1
        client=self.fake()
        with tempfile.TemporaryDirectory() as directory:
            summary=execute(config,Path(directory),client)
            self.assertEqual(summary['status'],'call_limit')
            self.assertEqual(client.calls,1)

    @patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-placeholder'})
    def test_anthropic_payload_and_empty_usage(self):
        client=DeepSeek('fake',protocol='anthropic',thinking='disabled',allow_empty=True)
        payload=dict(content=[],usage=dict(input_tokens=23,output_tokens=5),stop_reason='max_tokens')
        with patch.object(client.opener,'open',return_value=io.BytesIO(json.dumps(payload).encode())) as opened:
            self.assertEqual(client('q','c'),'')
        request=opened.call_args.args[0]
        self.assertTrue(request.full_url.endswith('/anthropic/v1/messages'))
        self.assertEqual(json.loads(request.data)['reasoning'],{'effort':'none'})
        self.assertEqual(client.last['usage']['prompt_tokens'],23)

    def test_bootstrap_uses_documents_not_question_count(self):
        rows=[dict(document_id='a',score=1)]*10+[dict(document_id='b',score=0)]
        result=cluster_ci(rows,'score',draws=100)
        self.assertEqual(result['macro_mean'],.5)
        self.assertEqual(result['documents'],2)

    def test_keyboard_interrupt_marks_run_incomplete(self):
        config=dict(backend='offline',data='data/toy_qa.jsonl',limit=1,
                    strategies=[dict(name='full',method='full')])
        with tempfile.TemporaryDirectory() as directory:
            with patch('token_budget_lab.experiment.extractive_answer',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute(config,Path(directory))
            result=json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(result['status'],'incomplete')

    @patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-placeholder'})
    def test_malformed_answer_preserves_known_usage(self):
        client=DeepSeek('fake')
        payload=dict(usage=dict(prompt_tokens=23,completion_tokens=5),choices=[])
        with patch.object(client.opener,'open',return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaises(IndexError):
                client('q','c')
        self.assertEqual(client.last['usage']['completion_tokens'],5)

    def test_optional_llmlingua_rejects_unpinned_weights(self):
        from token_budget_lab.llmlingua_adapter import LLMLingua2
        with self.assertRaisesRegex(ValueError,'revision'):
            LLMLingua2('example','main')

    def test_pilot_spans_ten_documents_without_api(self):
        config=json.loads((ROOT/'configs/deepseek_pilot.json').read_text())
        result=plan(config)
        self.assertEqual(result['documents'],10)
        self.assertEqual(result['planned_calls_without_failures'],40)
