import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab import ContextPlugin, HistoryArchive
from token_budget_lab.evidence_locator import EvidenceLocator, centered_span
from token_budget_lab.memory import encode
from token_budget_lab.retrieval import EvidenceIndex


class LocatorTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(events=[{'event':'scan_d-900','padding':'x'*4000,'status_code':206},
                                   {'event':'scan_d-43','status_code':202}])
        self.index = EvidenceLocator({'source':encode(self.source)})

    def test_sparse_identifier_and_shuffled_array_return_actual_location(self):
        result = self.index.find('source','event','scan_d-900',['status_code'])
        match = result['matches'][0]
        self.assertEqual(match['pointer'],'/events/0')
        self.assertEqual(match['field_pointers']['status_code'],'/events/0/status_code')
        self.assertEqual(match['fields'],{'status_code':206})
        self.assertFalse(match['record_complete'])
        self.assertNotIn('record',match)
        self.assertLess(len(encode(result)),1000)

    def test_search_returns_complete_matching_object_beyond_old_chunk_head(self):
        result = self.index.search('Find scan_d-43 status_code')
        self.assertEqual(result['results'][0]['record'],self.source['events'][1])
        self.assertEqual(result['results'][0]['pointer'],'/events/1')
        self.assertTrue(result['results'][0]['record_complete'])

    def test_duplicate_identifiers_preserve_all_candidates_and_count(self):
        rows = [{'event':'same','status_code':x} for x in range(5)]
        index = EvidenceLocator({'s':encode({'events':rows})})
        result = index.find('s','event','same',['status_code'],limit=3)
        self.assertEqual(result['match_count'],5)
        self.assertTrue(result['ambiguous'])
        self.assertTrue(result['has_more'])
        self.assertEqual([r['fields']['status_code'] for r in result['matches']],[0,1,2])

    def test_missing_exact_identifier_does_not_match_nearby_prefix(self):
        self.assertEqual(self.index.find('source','event','scan_d-9')['match_count'],0)
        result = self.index.search('scan_d-9 status_code')
        self.assertTrue(all(r['kind']=='text_excerpt' for r in result['results']))

    def test_pointer_escaping_and_missing_field_are_explicit(self):
        index = EvidenceLocator({'s':encode({'a/b~':[{'event':'abc','status/code~':503}]})})
        result = index.find('s','event','abc',['status/code~','absent'])['matches'][0]
        self.assertEqual(result['field_pointers']['status/code~'],'/a~1b~0/0/status~1code~0')
        self.assertNotIn('absent',result['fields'])

    def test_return_values_do_not_mutate_index_or_original_document(self):
        result = self.index.find('source','event','scan_d-43')
        result['matches'][0]['record']['status_code']=999
        self.assertEqual(self.index.find('source','event','scan_d-43')['matches'][0]['record']['status_code'],202)

    def test_duplicate_json_keys_are_not_silently_resolved(self):
        index = EvidenceLocator({'s':'{"event":"x","status_code":200,"status_code":503}'})
        with self.assertRaises(ValueError):
            index.find('s','event','x')

    def test_text_excerpt_contains_tail_anchor_and_exact_original_offsets(self):
        raw = 'background '*80+'needle_marker status_code=206 '+'background '*80
        index = EvidenceLocator({'s':raw})
        result = index.search('needle_marker status_code')
        row = next(r for r in result['results'] if 'needle_marker' in r['text'])
        self.assertEqual(row['text'],raw[row['offset']:row['end_offset']])
        self.assertLessEqual(len(row['text']),500)

    def test_versions_remain_distinct_without_automatic_latest_selection(self):
        index = EvidenceLocator({'old':encode({'event':'abc','status_code':200}),
                                 'new':encode({'event':'abc','status_code':503})})
        result = index.search('abc status_code')
        self.assertTrue(result['ambiguous'])
        self.assertEqual({r['source_id'] for r in result['results']},{'old','new'})

    def test_large_requested_field_is_not_clipped_into_fake_complete_json(self):
        row = self.index.find('source','event','scan_d-900',['padding','status_code'])['matches'][0]
        self.assertIn('padding',row['omitted_fields'])
        self.assertNotIn('padding',row['fields'])
        self.assertEqual(row['fields']['status_code'],206)

    def test_multiple_large_objects_have_bounded_response_with_true_match_count(self):
        rows=[{'event':'same',**{f'field{i}':'x'*100 for i in range(8)}} for _ in range(10)]
        result=EvidenceLocator({'s':encode({'events':rows})}).find('s','event','same')
        self.assertLess(len(encode(result)),4000)
        self.assertEqual(result['match_count'],10)
        self.assertTrue(result['ambiguous'])
        self.assertTrue(result['has_more'])

    def test_plugin_lookup_is_scoped_bounded_and_legacy_control_has_old_schema(self):
        rows = [dict(role='user',content='Read log.'),
            dict(role='assistant',content=None,tool_calls=[dict(id='c',type='function',function=dict(name='read_log',arguments='{}'))]),
            dict(role='tool',tool_call_id='c',content=encode(self.source)),dict(role='user',content='Find event.')]
        with tempfile.TemporaryDirectory() as tmp:
            archive = HistoryArchive(Path(tmp)/'a.db')
            try:
                plugin = ContextPlugin(archive)
                sid = plugin.before_model('t','s',rows).context_metadata['offloaded_source_ids'][0]
                args = dict(source_id=sid,match_key='event',match_value='scan_d-900',fields=['status_code'])
                self.assertEqual(plugin.invoke_context('t','s','context_find',args)['matches'][0]['fields']['status_code'],206)
                with self.assertRaises(ValueError):
                    plugin.invoke_context('other','s','context_find',args)
                for bad in ({'limit':4},{'fields':['x']*9},{'match_value':900}):
                    with self.assertRaises(ValueError):
                        plugin.invoke_context('t','s','context_find',{**args,**bad})
                control = ContextPlugin(archive,evidence_mode='lexical')
                self.assertNotIn('context_find',[x['function']['name'] for x in control.context_tools])
                with self.assertRaises(ValueError):
                    control.invoke_context('t','s','context_find',args)
            finally:
                archive.close()


if __name__=='__main__':
    unittest.main()
