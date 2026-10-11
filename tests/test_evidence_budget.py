import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.evidence_budget import (
    Chunk, EvidenceIndex, choose_budget, chunks_from_documents, pack_evidence,
    paragraph_spans, source_paragraph_covered)
from token_budget_lab.rag_experiment import execute, export, parse_answer, prepare
from token_budget_lab.deepseek import normalize_usage


class EvidenceBudgetTests(unittest.TestCase):
    def test_offsets_overlap_and_initials(self):
        text = 'Dr. J. R. Smith paid 3.14 dollars. ' + 'many words ' * 35
        chunks = chunks_from_documents({'doc': text}, 100, 25)
        covered = set()
        for c in chunks:
            self.assertEqual(c.text, text[c.start:c.end])
            self.assertLessEqual(len(c.text), 100)
            covered.update(range(c.start, c.end))
        self.assertEqual(covered, set(range(len(text.rstrip()))))
        self.assertTrue(chunks[0].text.startswith('Dr. J. R. Smith'))

    def test_paragraph_boundaries(self):
        text = '  first paragraph\n\n second\n\n'
        self.assertEqual([text[a:b] for a,b in paragraph_spans(text)], ['first paragraph','second'])

    def test_search_all_documents_and_empty_query(self):
        index = EvidenceIndex({'other': 'Bananas grow in Peru.', 'target': 'Cobalt is blue.'})
        self.assertEqual(index.search('cobalt')[0]['chunk'].document_id, 'target')
        self.assertEqual(index.search('the'), [])
        self.assertEqual(choose_budget('the', [], index)['budget'], 1800)

    def test_budget_depends_only_on_question_and_candidates(self):
        index = EvidenceIndex({'doc': 'Cobalt has atomic number 27.'})
        matches = index.search('Cobalt atomic number')
        self.assertEqual(choose_budget('Cobalt atomic number', matches, index)['tier'], 'small')
        self.assertEqual(choose_budget('Cobalt compare nickel', matches, index)['tier'], 'large')
        self.assertEqual(choose_budget('Cobalt missingword', matches, index)['tier'], 'large')

    def test_whole_chunk_budget_and_duplicate_intervals(self):
        a = Chunk('a','d',0,90,'x'*90)
        b = Chunk('b','d',10,100,'x'*90)
        c = Chunk('c','other',0,40,'y'*40)
        candidates = [dict(chunk=x) for x in (a,b,c)]
        packed = pack_evidence(candidates, 190)
        self.assertEqual(packed['chunks'], [a,c])
        self.assertLessEqual(len(packed['context']), 190)
        self.assertEqual(pack_evidence(candidates, 20)['chunks'], [])

    def test_gold_paragraph_union_is_evaluation_only(self):
        text = 'abcdefghij'
        row = dict(document_id='d', source_paragraph_sha256=hashlib.sha256(text.encode()).hexdigest())
        parts = [Chunk('a','d',0,6,text[:6]),Chunk('b','d',4,10,text[4:])]
        self.assertTrue(source_paragraph_covered(row, {'d':text}, parts))
        self.assertFalse(source_paragraph_covered(row, {'d':text}, parts[:1]))

    def test_strict_response_and_citation_validation(self):
        value = '{"answer":"27","citations":["E1"],"abstain":false}'
        self.assertEqual(parse_answer(value, {'E1'})['answer'], '27')
        for raw in (value.replace('E1','E2'), value.replace('false','true'), '```json\n'+value+'\n```'):
            with self.assertRaises(ValueError):
                parse_answer(raw, {'E1'})
        self.assertTrue(parse_answer('{"answer":"","citations":[],"abstain":true}', set())['abstain'])

    def test_anthropic_total_includes_cache_reads_and_writes(self):
        usage = dict(input_tokens=20, cache_read_input_tokens=128,
                     cache_creation_input_tokens=64, output_tokens=7, prompt_tokens=20)
        fixed = normalize_usage(usage, 'anthropic')
        self.assertEqual(fixed['prompt_tokens'], 212)
        self.assertEqual(fixed['prompt_cache_hit_tokens'], 128)
        self.assertEqual(fixed['prompt_cache_miss_tokens'], 84)
        self.assertEqual(fixed['input_tokens'], 20)
        self.assertEqual(normalize_usage(fixed,'anthropic'), fixed)
        self.assertEqual(normalize_usage(dict(prompt_tokens=212,completion_tokens=7,
                                              prompt_cache_hit_tokens=128), 'chat')['prompt_tokens'], 212)
        with self.assertRaises(RuntimeError):
            normalize_usage(dict(usage, cache_read_input_tokens=-1), 'anthropic')


class RagRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        paragraph = 'Cobalt has atomic number 27.'
        self.data = self.root/'data.json'
        self.data.write_text(json.dumps(dict(documents={'d':dict(context=paragraph,split='dev')},
            questions=[dict(id=f'q{i}', document_id='d', query='What is the atomic number of cobalt?',
                            answers=['27'], source_paragraph_sha256=hashlib.sha256(paragraph.encode()).hexdigest())
                       for i in range(2)])), encoding='utf-8')
        self.config = dict(data=str(self.data), split='dev', seed=7, backend='offline', baseline='fixed',
                           strategies=[dict(name='fixed',method='fixed',budget=1800),
                                       dict(name='adaptive',method='adaptive'),dict(name='shuffled',method='shuffled')])

    def tearDown(self):
        self.temp.cleanup()

    def paid(self, max_calls=6):
        return dict(self.config, backend='deepseek', model='fake', protocol='chat', thinking='disabled',
                    max_tokens=100, max_calls=max_calls)

    def test_offline_has_no_generated_accuracy_and_shuffled_same_multiset(self):
        rows, _, index, manifest, _ = prepare(self.config)
        budgets = [choose_budget(r['query'], index.search(r['query']), index)['budget'] for r in rows]
        self.assertEqual(sorted(budgets), sorted(manifest['shuffled_budgets'].values()))
        result = execute(self.config, self.root/'offline')
        self.assertTrue(all(r['exact_match'] is None and r['quality_screen']=='proxy_only' for r in result['runs']))

    def test_reference_changes_do_not_change_budget_or_selected_evidence(self):
        execute(self.config, self.root/'gold_one')
        bundle = json.loads(self.data.read_text(encoding='utf-8'))
        for q in bundle['questions']:
            q['answers'] = ['Cobalt']
        self.data.write_text(json.dumps(bundle), encoding='utf-8')
        execute(self.config, self.root/'gold_two')
        def selections(name):
            rows = [json.loads(l) for l in (self.root/name/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
            return [(r['key'], r['budget_characters'], r['chunk_ids']) for r in rows]
        self.assertEqual(selections('gold_one'), selections('gold_two'))

    def test_export_recounts_raw_anthropic_usage_without_rewriting_journal(self):
        class Client:
            last = {}
            def __call__(self, query, context):
                self.last = dict(usage=dict(input_tokens=30,cache_read_input_tokens=64,
                                           output_tokens=8,prompt_tokens=30,completion_tokens=8))
                return '{"answer":"27","citations":["E1"],"abstain":false}'
        config = dict(self.paid(), protocol='anthropic')
        execute(config, self.root/'raw', Client())
        before = (self.root/'raw'/'requests.jsonl').read_bytes()
        export(self.root/'raw', self.root/'public')
        self.assertEqual(before, (self.root/'raw'/'requests.jsonl').read_bytes())
        summary = json.loads((self.root/'public'/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(sum(r['prompt_tokens'] for r in summary['runs']), 6*94)
        self.assertNotIn('raw_answer', (self.root/'public'/'per_question.csv').read_text(encoding='utf-8').splitlines()[0])

    def test_resume_never_repeats_completed_paid_calls(self):
        class Client:
            last = {}
            calls = 0
            def __call__(self, query, context):
                self.calls += 1
                self.last = dict(usage=dict(prompt_tokens=30,completion_tokens=8), finish_reason='stop')
                return '{"answer":"27","citations":["E1"],"abstain":false}'
        client = Client()
        config = self.paid()
        result = execute(config, self.root/'paid', client)
        self.assertEqual(client.calls, 6)
        self.assertTrue(all(r['exact_match']==1 for r in result['runs']))
        execute(config, self.root/'paid', client, resume=True)
        self.assertEqual(client.calls, 6)
        with self.assertRaises(ValueError):
            execute(dict(config, seed=8), self.root/'paid', client, resume=True)

    def test_call_cap_and_malformed_paid_response_accounting(self):
        class Client:
            last = {}
            calls = 0
            def __call__(self, query, context):
                self.calls += 1
                self.last = dict(usage=dict(prompt_tokens=30,completion_tokens=8))
                return 'invalid JSON'
        client = Client()
        with self.assertRaises(RuntimeError):
            execute(self.paid(max_calls=1), self.root/'cap', client)
        summary = json.loads((self.root/'cap'/'summary.json').read_text())
        self.assertEqual(client.calls, 1)
        self.assertEqual(sum(r['prompt_tokens'] for r in summary['runs']), 30)
        self.assertEqual(sum(r['errors'] for r in summary['runs']), 1)

    def test_unresolved_attempt_is_not_retried(self):
        execute(self.config, self.root/'pending')
        with (self.root/'pending'/'requests.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(event='start', key='unresolved'))+'\n')
        with self.assertRaises(RuntimeError):
            execute(self.config, self.root/'pending', resume=True)


if __name__ == '__main__':
    unittest.main()
