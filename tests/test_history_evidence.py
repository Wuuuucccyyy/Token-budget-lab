import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.history_evidence import plan_evidence
from token_budget_lab.memory import HistoryArchive, MemoryPlugin, split_turns, validate_messages
from token_budget_lab.memory_agent import run_agent
from token_budget_lab.long_context_tasks import build_long_tasks
from token_budget_lab.tool_tasks import ToolSandbox, assess, TOOLS
from token_budget_lab.scripted_policy import ScriptedPolicy
from token_budget_lab.tool_experiment import summarize


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

    def test_recovery_schema_removed_after_full_restore(self):
        class Transport:
            last={}
            count=0
            def __call__(self,messages,tools):
                self.count+=1
                names={t['function']['name'] for t in tools}
                if self.count==1:
                    self_test.assertIn('history_restore_all',names)
                    message=dict(role='assistant',content=None,tool_calls=[dict(id='all',type='function',
                        function=dict(name='history_restore_all',arguments='{}'))])
                else:
                    self_test.assertEqual(names,set())
                    message=dict(role='assistant',content='done')
                return dict(message=message,usage=dict(prompt_tokens=20,completion_tokens=4))
        self_test=self
        rows=snapshots('Alpha config 最新 timeout_ms？')
        # Unrelated old tool output creates a genuine compression opportunity.
        rows[7]['content']='background '*800
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'history.sqlite3')
            try:
                result=run_agent(rows,Transport(),MemoryPlugin(archive,budget=5000,recent_turns=1),'t','s')
            finally: archive.close()
        self.assertEqual(result['status'],'ok')
        self.assertEqual([s['recovery_tools_enabled'] for s in result['steps']],[True,False])
        validate_messages(result['messages'])

    def test_short_history_has_no_recovery_tools(self):
        class Transport:
            last={}
            def __call__(self,messages,tools):
                self_test.assertEqual(tools,[])
                return dict(message=dict(role='assistant',content='ok'))
        self_test=self
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'h.sqlite3')
            try: run_agent([dict(role='user',content='short')],Transport(),MemoryPlugin(archive),'t','s')
            finally: archive.close()

    def test_active_write_does_not_replace_the_historical_read(self):
        rows=snapshots('使用最近一次读取的 Alpha config timeout_ms 加 quota，写入当前配置。')
        rows[7]['content']='background '*800
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'h.sqlite3')
            plugin=MemoryPlugin(archive,budget=5000,recent_turns=1)
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


class LengthEvaluationTests(unittest.TestCase):
    def test_length_variants_match_and_labels_do_not_enter_history(self):
        bundle=build_long_tasks()
        self.assertEqual(len(bundle['episodes']),72)
        dev={e['base_id'] for e in bundle['episodes'] if e['split']=='dev'}
        test={e['base_id'] for e in bundle['episodes'] if e['split']=='test'}
        self.assertFalse(dev & test)
        base=[e for e in bundle['episodes'] if e['base_id']=='temporal-00']
        self.assertEqual(len({tuple(c['query'] for c in e['checkpoints']) for e in base}),1)
        sizes=[]
        with tempfile.TemporaryDirectory() as directory:
            for e in base:
                rows,_=ToolSandbox(Path(directory)/e['id'],e).seed_history(e)
                sizes.append(len(json.dumps(rows)))
            altered=copy.deepcopy(base[-1])
            for c in altered['checkpoints']:c['answer']='SECRET';c['timeout_ms']=123456
            other,_=ToolSandbox(Path(directory)/'altered',altered).seed_history(altered)
            self.assertEqual(rows,other)
        self.assertLess(sizes[0],sizes[1]);self.assertLess(sizes[1],sizes[2])

    def test_complex_long_tasks_execute_with_selected_original_evidence(self):
        examples=[e for e in build_long_tasks()['episodes'] if e['context_length']=='long' and e['base_id'].endswith('-00')]
        with tempfile.TemporaryDirectory() as directory:
            archive=HistoryArchive(Path(directory)/'history.sqlite3')
            try:
                for e in examples:
                    sandbox=ToolSandbox(Path(directory)/e['id'],e)
                    history,_=sandbox.seed_history(e)
                    plugin=MemoryPlugin(archive,budget=9000,recent_turns=1)
                    policy=ScriptedPolicy()
                    for number,c in enumerate(e['checkpoints']):
                        history.append(dict(role='user',content=c['query']))
                        start=len(sandbox.trace);before=sandbox.file_hashes()
                        result=run_agent(history,policy,plugin,'test',e['id'],max_steps=6,extra_tools=TOOLS,handlers=sandbox.handlers())
                        self.assertEqual(result['steps'][0]['history_mode'],'compressed',(e['id'],number,result['steps'][0]['history_reason']))
                        score=assess(c,result,sandbox,before,start,e['initial_config'])
                        self.assertTrue(score['task_success'],(e['id'],number,score))
                        history=result['messages']
            finally:archive.close()

    def test_compressed_subset_includes_failures_and_matches_baseline(self):
        records=[]
        for mode in ('full','recover'):
            for episode,compressed in [('long',mode=='recover'),('short',False)]:
                metadata=dict(episode_id=episode,base_id=episode,context_length=episode,kind='temporal',mode=mode)
                records.append(dict(event='start',checkpoint=0,**metadata))
                records.append(dict(event='response',history_mode='compressed' if compressed else 'full',history_reason='test',
                    usage=dict(prompt_tokens=40 if compressed else 100,completion_tokens=5),**metadata))
                success=not compressed
                records.append(dict(event='task_result',checkpoint=0,task_success=success,answer_correct=success,format_ok=True,
                    state_correct=success,required_tools_completed=True,constraint_violation=False,answer='ok',status='ok',
                    initial_history_mode='compressed' if compressed else 'full',initial_sent_units=100,
                    source_messages_retained=True,business_tool_calls=0,end_to_end_seconds=1,compressed_requests=int(compressed),**metadata))
        manifest=dict(config=dict(backend='deepseek',modes=['full','recover'],seed=1),episode_ids=['short','long'],
            checkpoints=2,expected_episode_checkpoints={'short':1,'long':1})
        summary=summarize(records,manifest)
        subset=summary['compressed_subset'][0]
        self.assertEqual(subset['episodes'],1)
        self.assertEqual(subset['episode_success_rate'],0)
        self.assertEqual(subset['matching_full_episode_success_rate'],1)
        self.assertEqual(subset['matching_full_total_tokens'],105)
        self.assertEqual(subset['total_tokens'],45)


if __name__=='__main__':unittest.main()
