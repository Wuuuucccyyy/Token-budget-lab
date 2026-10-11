import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.context_plugin import ContextPlugin, source_id, tool_sources
from token_budget_lab.memory import HistoryArchive, encode, scope_key, validate_messages
from token_budget_lab.memory_agent import run_agent
from token_budget_lab.provenance import HistoricalWriteGuard, ProvenanceRejected
from token_budget_lab.tool_tasks import TOOLS
from helpers import Script, BUSINESS


def add_result(messages, name, value, number):
    messages += [dict(role='user', content='Read and retain '+name),
        dict(role='assistant', content=None, tool_calls=[dict(id='c'+str(number), type='function',
             function=dict(name=name, arguments='{}'))]),
        dict(role='tool', tool_call_id='c'+str(number), content=encode(value)),
        dict(role='assistant', content='Read completed.')]


def history(query='截至 2026-10-06，选最近的 Alpha config 快照，取 timeout_ms 写入配置。'):
    messages = [dict(role='system', content='Use actual source evidence and business tools.')]
    add_result(messages, 'read_config', dict(observed_at='2026-10-01', config=dict(module='Alpha', timeout_ms=615, quota=35)), 1)
    add_result(messages, 'read_config', dict(observed_at='2026-10-05', config=dict(module='Alpha', timeout_ms=985, quota=35)), 2)
    add_result(messages, 'read_log', dict(events=[dict(event=i, detail='background '*90) for i in range(100)]), 3)
    return messages+[dict(role='user', content=query)]


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = HistoryArchive(Path(self.temp.name)/'a.db')
        self.plugin = ContextPlugin(self.archive, budget=2400)

    def tearDown(self):
        self.archive.close()
        self.temp.cleanup()

    def test_raw_archive_and_roles_survive_offload(self):
        rows = history(); original = copy.deepcopy(rows)
        p = self.plugin.before_model('t','s',rows)
        self.assertEqual(rows, original)
        self.assertEqual(self.archive.read(scope_key('t','s')), rows)
        validate_messages(p.messages)
        self.assertLess(p.sent_units, p.baseline_units/3)
        self.assertTrue(p.context_metadata['offloaded_source_ids'])
        self.assertNotIn('background '*90, encode(p.messages))

    def test_new_huge_result_does_not_change_previously_sent_prefix(self):
        rows = history(); first = self.plugin.before_model('t','s',rows)
        rows += [dict(role='assistant',content=None,tool_calls=[dict(id='new',type='function',
                 function=dict(name='read_log',arguments='{}'))]),
                 dict(role='tool',tool_call_id='new',content=encode(dict(text='large '*12000)))]
        second = self.plugin.before_model('t','s',rows)
        self.assertEqual(second.messages[:len(first.messages)], first.messages)
        self.assertLess(second.sent_units-first.sent_units, 1000)
        self.assertFalse(second.context_metadata['selection_stage_changed'])
        self.assertEqual(second.context_metadata['common_prefix_messages'], len(first.messages))
        self.assertEqual(second.mode,'compressed')

    def test_pointer_read_exact_page_and_session_isolation(self):
        rows = history(); p = self.plugin.before_model('t','s',rows)
        sid = p.context_metadata['offloaded_source_ids'][0]
        result = self.plugin.invoke_context('t','s','context_read',dict(source_id=sid,pointer='/events/50',limit=2000))
        self.assertEqual(json.loads(result['text'])['event'],50)
        result = self.plugin.invoke_context('t','s','context_read',dict(source_id=sid,limit=64))
        raw = tool_sources(rows)[sid]['message']['content']
        self.assertEqual(result['text'],raw[:64]); self.assertEqual(result['next_offset'],64)
        with self.assertRaises(ValueError):
            self.plugin.invoke_context('t','different','context_read',dict(source_id=sid))
        for bad in (dict(limit=2001),dict(offset=-1),dict(pointer='/events/-1')):
            with self.assertRaises((ValueError,KeyError,IndexError)):
                self.plugin.invoke_context('t','s','context_read',dict(source_id=sid,**bad))

    def test_persisted_tool_view_is_identical_after_restart(self):
        rows=history(); a=self.plugin.before_model('t','s',rows)
        other=ContextPlugin(self.archive,budget=2400)
        b=other.before_model('t','s',rows)
        self.assertEqual(a.messages,b.messages)

    def test_unknown_query_keeps_small_facts_and_offloads_logs(self):
        rows=history('Explain a relationship that the planner does not recognize.')
        p=self.plugin.before_model('t','s',rows)
        self.assertIn(rows[3],p.messages); self.assertIn(rows[7],p.messages)
        self.assertLess(p.sent_units,p.baseline_units)

    def test_soft_budget_never_discards_current_user_constraint(self):
        rows=history()+[dict(role='user',content='explicit constraint '*400)]
        p=self.plugin.before_model('t','s',rows)
        self.assertIn(rows[-1],p.messages)
        self.assertTrue(p.context_metadata['soft_budget_exceeded'])

    def test_search_is_bounded_and_excludes_retrieval_echoes(self):
        rows=history(); add_result(rows,'context_read',dict(text='unique_echo_secret'),4)
        self.plugin.before_model('t','s',rows)
        result=self.plugin.invoke_context('t','s','context_search',dict(query='background',limit=3))
        self.assertLess(len(encode(result)),2200)
        result=self.plugin.invoke_context('t','s','context_search',dict(query='unique_echo_secret'))
        self.assertFalse(result['results'])

    def test_actual_transport_prefix_and_schemas_stay_stable_with_read_guard(self):
        read=next(t for t in TOOLS if t['function']['name']=='read_report')
        script=Script([[('read_report',{})],[('finish',{})],'done'])
        result=run_agent(history(),script,self.plugin,'t','s',max_steps=4,extra_tools=[read,BUSINESS],
            handlers={'read_report':lambda:dict(report=None),'finish':lambda:dict(done=True)},snapshot_read_tools=['read_report'])
        self.assertEqual(result['status'],'ok')
        for n in range(1,3):
            self.assertEqual(script.requests[n][0][:len(script.requests[n-1][0])],script.requests[n-1][0])
            self.assertEqual(script.requests[n][1],script.requests[0][1])
        self.assertFalse(any(t['function']['name'].startswith('history_') for t in script.requests[0][1]))

    def test_partial_mutation_failure_invalidates_read_snapshot(self):
        read=next(t for t in TOOLS if t['function']['name']=='read_report')
        values=[]; state=[1]
        def get(): values.append(state[0]); return dict(report=state[0])
        def mutate(): state[0]=2; raise RuntimeError('partial write')
        script=Script([[('read_report',{})],[('finish',{})],[('read_report',{})],'done'])
        run_agent(history(),script,self.plugin,'t','s',max_steps=4,extra_tools=[read,BUSINESS],
            handlers={'read_report':get,'finish':mutate},snapshot_read_tools=['read_report'])
        self.assertEqual(values,[1,2])


class ProvenanceTests(unittest.TestCase):
    def proof(self,g,value,**kwargs):
        d=g.demand
        return dict(timeout_ms=value,evidence=dict(operation=d['operation'],source_ids=[s[0] for s in d['source_groups']],
                    fields=d['fields'],constant=d['constant'],**kwargs))

    def test_as_of_old_value_rejected_and_correct_value_accepted(self):
        g=HistoricalWriteGuard(history())
        with self.assertRaises(ProvenanceRejected): g.validate('write_config',self.proof(g,615))
        self.assertEqual(g.validate('write_config',self.proof(g,985)),dict(timeout_ms=985))
        old=source_id(3,history()[3]); args=self.proof(g,615); args['evidence']['source_ids']=[old]
        with self.assertRaises(ProvenanceRejected): g.validate('write_config',args)

    def test_difference_sum_and_offset_require_exact_operation(self):
        for query,expected in [
            ('截至 2026-10-04 与截至 2026-10-06，各选最近 Alpha config 快照，计算后一个 timeout_ms 减前一个 timeout_ms。',370),
            ('最近一次读取的 Alpha config 快照取 timeout_ms 和 quota 相加写入。',1020),
            ('首次读取的 Alpha config 快照取 timeout_ms 加 5 写入。',620)]:
            g=HistoricalWriteGuard(history(query)); self.assertIsNone(g.error)
            self.assertEqual(g.validate('write_config',self.proof(g,expected))['timeout_ms'],expected)
            args=self.proof(g,expected);args['evidence']['constant']+=1
            with self.assertRaises(ProvenanceRejected):g.validate('write_config',args)

    def test_conflicting_same_date_and_unsupported_arithmetic_reject(self):
        rows=history()[:-1]
        add_result(rows,'read_config',dict(observed_at='2026-10-05',config=dict(module='Alpha',timeout_ms=777,quota=35)),4)
        g=HistoricalWriteGuard(rows+[history()[-1]])
        self.assertEqual(g.error,'missing_or_conflicting_numeric_field')
        g=HistoricalWriteGuard(history('取最近 Alpha config 的 timeout_ms 乘 2 写入。'))
        self.assertEqual(g.error,'unsupported_arithmetic_requirement')

    def test_undated_snapshot_does_not_silently_select_older_version(self):
        rows=history()[:-1]
        add_result(rows,'read_config',dict(config=dict(module='Alpha',timeout_ms=1111)),4)
        g=HistoricalWriteGuard(rows+[history()[-1]])
        self.assertEqual(g.error,'undated_snapshot_cannot_establish_version_order')

    def test_current_task_read_does_not_replace_historical_contract(self):
        rows=history();g=HistoricalWriteGuard(rows)
        self.assertEqual(g.validate('write_config',self.proof(g,985))['timeout_ms'],985)
        self.assertEqual(len(g.facts),2)

    def test_rejected_write_never_invokes_handler_then_model_can_correct(self):
        rows=history();g=HistoricalWriteGuard(rows);writes=[]
        write=next(t for t in TOOLS if t['function']['name']=='write_config')
        script=Script([[('write_config',self.proof(g,615))],[('write_config',self.proof(g,985))],'{"answer":"985"}'])
        result=run_agent(rows,script,None,'t','s',mode='full',memory_tools=False,max_steps=3,
            extra_tools=[write],handlers={'write_config':lambda timeout_ms:writes.append(timeout_ms) or dict(written=True)},write_guard=g)
        self.assertEqual(writes,[985]);self.assertEqual(result['status'],'ok')
        self.assertEqual(json.loads(result['messages'][len(rows)+1]['content'])['error'],'provenance_rejected')
        self.assertTrue(any('source_ids' in m.get('content','') for m in script.requests[0][0] if isinstance(m.get('content'),str)))
        self.assertFalse(any(m.get('content')==g.notice() for m in result['messages']))


if __name__=='__main__':unittest.main()
