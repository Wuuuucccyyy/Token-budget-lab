import json
import tempfile
import unittest
from pathlib import Path
from token_budget_lab.history_evidence import plan_evidence
from token_budget_lab.memory import HistoryArchive, split_turns
from token_budget_lab.context_plugin import ContextPlugin

def snapshots(query):
    messages = [dict(role='system', content='Use dated observations, never infer absent values.')]
    for i, (module, when, value) in enumerate([
        ('Alpha', '2026-10-01', 100), ('Beta', '2026-10-03', 800),
        ('Alpha', '2026-10-05', 200), ('Alpha', '2026-10-10', 900)]):
        messages += [dict(role='user', content=f'Read {module} config.'),
            dict(role='assistant', content=None, tool_calls=[dict(id=f'r{i}', type='function',
                function=dict(name='read_config', arguments='{}'))]),
            dict(role='tool', tool_call_id=f'r{i}', content=json.dumps(dict(observed_at=when,
                config=dict(module=module, timeout_ms=value, quota=30)))),
            dict(role='assistant', content='Read completed.')]
    messages.append(dict(role='user', content=query))
    return messages


class EvidenceTests(unittest.TestCase):
    def test_as_of_excludes_later_versions_and_other_entities(self):
            messages=snapshots('截至 2026-10-06，Alpha config 的 timeout_ms 是多少？')
            plan=plan_evidence(messages[-1]['content'],split_turns(messages))
            self.assertTrue(plan.supported)
            self.assertEqual(plan.turn_ids, {'T9'})
            self.assertEqual(plan.details['temporal'], 'as_of')

    def test_compare_keeps_both_cutoff_sources(self):
            messages=snapshots('比较截至 2026-10-04 与截至 2026-10-06，Alpha config timeout_ms 的差值。')
            plan=plan_evidence(messages[-1]['content'],split_turns(messages))
            self.assertEqual(plan.turn_ids, {'T1','T9'})

    def test_unknown_field_retains_all_subject_observations(self):
            messages=snapshots('Alpha billing_email 未记录时回答 UNKNOWN，Beta 的联系人不能替代。')
            plan=plan_evidence(messages[-1]['content'],split_turns(messages))
            self.assertTrue(plan.supported)
            self.assertEqual(plan.turn_ids, {'T1','T9','T13'})
            self.assertEqual(plan.details['missing_fields'], ['billing_email'])

    def test_ambiguous_subject_and_unsupported_dates_fall_back(self):
            for query, reason in [('config timeout_ms 是多少？','ambiguous_subject'),
                                  ('在 2026-10-06 之前 Alpha config timeout_ms 是多少？','unsupported_temporal_expression')]:
                plan=plan_evidence(query,split_turns(snapshots(query)))
                self.assertFalse(plan.supported)
                self.assertEqual(plan.reason,reason)

    def test_equal_date_conflicts_are_both_preserved(self):
            messages=snapshots('Alpha config timeout_ms 最新值？')
            messages[15]['content']=json.dumps(dict(observed_at='2026-10-05',config=dict(module='Alpha',timeout_ms=900)))
            plan=plan_evidence(messages[-1]['content'],split_turns(messages))
            self.assertEqual(plan.turn_ids,{'T9','T13'})

    def test_total_is_a_field_not_a_request_for_every_record(self):
            messages=snapshots('report.json 的 total 是多少？')[:-1]
            messages += [dict(role='user',content='Read report.'),
                dict(role='assistant',content=None,tool_calls=[dict(id='report',type='function',function=dict(name='read_report',arguments='{}'))]),
                dict(role='tool',tool_call_id='report',content='{"report":{"total":"12.50","currency":"USD"}}'),
                dict(role='user',content='report.json 的 total 是多少？')]
            plan=plan_evidence(messages[-1]['content'],split_turns(messages))
            self.assertTrue(plan.supported)
            self.assertEqual(plan.details['temporal'],'latest')

    def test_active_write_does_not_replace_the_historical_read(self):
            rows=snapshots('使用最近一次读取的 Alpha config timeout_ms 加 quota，写入当前配置。')
            rows[7]['content']='background '*800
            with tempfile.TemporaryDirectory() as directory:
                archive=HistoryArchive(Path(directory)/'h.sqlite3')
                plugin=ContextPlugin(archive,budget=5000,recent_turns=1)
                try:
                    original=plugin.before_model('t','s',rows)
                    rows += [dict(role='assistant',content=None,tool_calls=[dict(id='new',type='function',
                        function=dict(name='write_config',arguments='{"timeout_ms":930}'))]),
                        dict(role='tool',tool_call_id='new',content='{"written":true,"config":{"module":"Alpha","timeout_ms":930,"quota":30,"observed_at":"2026-10-11"}}')]
                    after=plugin.before_model('t','s',rows)
                    self.assertEqual(original.restored_turn_ids,after.restored_turn_ids)
                    self.assertIn(rows[15],after.messages)
                    self.assertIn(rows[-1],after.messages)
                finally:archive.close()
