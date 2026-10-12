import copy
import json
import tempfile
import unittest
from pathlib import Path

from helpers import Script
from token_budget_lab import ContextPlugin, FreshWriteGuard, HistoryArchive, WriteTransform, run_agent
from token_budget_lab.action_experiment import plan
from token_budget_lab.action_tasks import execution_contract
from token_budget_lab.context_plugin import source_id
from token_budget_lab.memory import encode, scope_key
from token_budget_lab.tool_tasks import TOOLS


def guard():
    return FreshWriteGuard([WriteTransform('read_log',{'name':'scan_c'},'write_report','total',
        {'kind':'unique_event','array':'/events','match_key':'event','match_value':'scan_c-901','field':'status_code'},
        fixed_arguments={'currency':'USD'})])


def body(code=206):
    return {'events':[{'event':'scan_c-1','status_code':200},{'event':'scan_c-901','status_code':code}], 'noise':'background '*500}


def observe(g,value,args=None):
    message={'role':'tool','tool_call_id':'1-0','content':encode(value)}
    g.observe('read_log',args or {'name':'scan_c'},value,message,3,True)
    return message,source_id(3,message)


def history(value):
    return [{'role':'system','content':'Use tools'}, {'role':'user','content':'Actual read, format, write and check.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'1-0','type':'function','function':{'name':'read_log','arguments':encode({'name':'scan_c'})}}]},
        {'role':'tool','tool_call_id':'1-0','content':encode(value)}]


class EagerEvidenceTests(unittest.TestCase):
    def test_projection_contains_original_field_and_actual_location_not_transformed_answer(self):
        g=guard();_,sid=observe(g,body());v=g.evidence_views()[sid][0]
        self.assertEqual(v['original_value'],206)
        self.assertEqual(v['source_location'],'/events/1/status_code')
        self.assertEqual(v['selection_status'],'unique')
        self.assertNotIn('expected_value',v)
        self.assertNotIn('206.00',encode(v))

    def test_missing_and_duplicate_status_comes_from_entire_actual_array(self):
        for events,status in [([], 'missing'),([{'event':'scan_c-901','status_code':206}]*2,'ambiguous')]:
            g=guard();_,sid=observe(g,dict(events=events));v=g.evidence_views()[sid][0]
            self.assertEqual(v['selection_status'],status)
            self.assertTrue(v['selection_complete']);self.assertNotIn('original_value',v)

    def test_unavailable_fields_fall_back_without_inventing_absence(self):
        for value in [body('huge '*1000), {'events':[{'event':'scan_c-901'}]}]:
            g=guard();_,sid=observe(g,value);v=g.evidence_views()[sid][0]
            self.assertEqual(v['selection_status'],'unavailable')
            self.assertFalse(v['selection_complete']);self.assertLess(len(encode(v)),1000)

    def test_old_or_wrong_file_does_not_produce_eager_evidence(self):
        g=guard();self.assertEqual(g.evidence_views(),{})
        observe(g,body(),{'name':'scan_a'});self.assertEqual(g.evidence_views(),{})

    def test_fixed_text_record_recovers_tail_without_paging_or_formatting(self):
        g=FreshWriteGuard([WriteTransform('read_log',{'name':'scan_c'},'write_report','total',
            {'kind':'text_record','pointer':'/log_text','marker':'TAIL_MARKER','field':'status_code'})])
        _,sid=observe(g,{'log_text':'noise '*1500+'TAIL_MARKER status_code=206;'})
        view=g.evidence_views()[sid][0]
        self.assertEqual(view['original_value'],'206');self.assertEqual(view['selection_status'],'unique')
        self.assertGreater(view['source_location']['offset'],8000)

    def test_view_preserves_raw_archive_prefix_and_scope(self):
        g=guard();value=body();_,sid=observe(g,value);rows=history(value);original=copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as tmp:
            a=HistoryArchive(Path(tmp)/'a.db')
            try:
                p=ContextPlugin(a,budget=2000)
                first=p.before_model('t','s',rows,evidence_views=g.evidence_views())
                rows.append({'role':'assistant','content':'done'})
                second=p.before_model('t','s',rows,evidence_views=g.evidence_views())
                self.assertEqual(second.messages[:len(first.messages)],first.messages)
                self.assertEqual(a.read(scope_key('t','s'))[:4],original)
                self.assertEqual(first.context_metadata['eager_evidence_source_ids'],[sid])
                self.assertEqual(json.loads(first.messages[3]['content'])['requested_evidence'][0]['original_value'],206)
                with self.assertRaises(ValueError):p.invoke_context('t','other','context_read',{'source_id':sid})
            finally:a.close()

    def test_source_fingerprint_mismatch_is_rejected_before_transport(self):
        g=guard();value=body();_,sid=observe(g,value);views=g.evidence_views();views[sid][0]['result_sha256']='fake'
        with tempfile.TemporaryDirectory() as tmp:
            a=HistoryArchive(Path(tmp)/'a.db')
            try:
                p=ContextPlugin(a)
                with self.assertRaises(ValueError):p.before_model('t','s',history(value),evidence_views=views)
            finally:a.close()

    def test_late_projection_does_not_rewrite_already_sent_source_view(self):
        g=guard();value=body();observe(g,value)
        with tempfile.TemporaryDirectory() as tmp:
            a=HistoryArchive(Path(tmp)/'a.db')
            try:
                p=ContextPlugin(a);old=p.before_model('t','s',history(value))
                new=p.before_model('t','s',history(value),evidence_views=g.evidence_views())
                self.assertEqual(new.messages,old.messages)
            finally:a.close()

    def test_agent_sends_fresh_original_field_after_actual_read_without_extra_recovery_call(self):
        rows=history(body())[:2];script=Script([[('read_log',{'name':'scan_c'})],
            [('write_report',{'total':'206.00','currency':'USD'})],[('run_checks',{})],'{"answer":"206.00"}'])
        writes=[];events=[]
        with tempfile.TemporaryDirectory() as tmp:
            a=HistoryArchive(Path(tmp)/'a.db')
            try:
                result=run_agent(rows,script,ContextPlugin(a),'t','s',max_steps=10,extra_tools=TOOLS,
                    handlers={'read_log':lambda name:body(),'write_report':lambda **v:writes.append(v) or {'written':True},'run_checks':lambda:{'passed':True}},
                    write_guard=guard(),eager_evidence=True,after_response=lambda r:events.append('paid'),after_tools=lambda r:events.append('tool'))
                view=next(json.loads(m['content']) for m in script.requests[1][0] if m['role']=='tool')
                self.assertEqual(view['requested_evidence'][0]['original_value'],206)
                self.assertEqual(result['status'],'ok');self.assertEqual(result['api_calls'],4)
                self.assertEqual(result['recovery_tool_calls'],0);self.assertEqual(len(writes),1)
                self.assertEqual(events[:2],['paid','tool'])
                self.assertEqual(json.loads(result['messages'][3]['content']),body())
            finally:a.close()

    def test_seventh_final_response_is_allowed_and_metered_under_ten_step_limit(self):
        value=body();message,sid=observe(guard(),value)
        responses=[[('read_log',{'name':'scan_c'})],
            [('context_find',{'source_id':sid,'match_key':'marker','match_value':'not_json'})],
            [('context_read',{'source_id':sid,'offset':0,'limit':200})],
            [('context_read',{'source_id':sid,'offset':200,'limit':200})],
            [('write_report',{'total':'206.00','currency':'USD'})],[('run_checks',{})],'{"answer":"206.00"}']
        for cap,status,calls in [(6,'step_limit',6),(10,'ok',7)]:
            with tempfile.TemporaryDirectory() as tmp:
                a=HistoryArchive(Path(tmp)/'a.db')
                try:
                    contract=execution_contract({'required':[{'id':'r','tool':'read_log','arguments':{'name':'scan_c'}}]})
                    result=run_agent(history(value)[:2],Script(responses),ContextPlugin(a),'t','s',max_steps=cap,max_recovery_rounds=3,
                        extra_tools=TOOLS,handlers={'read_log':lambda name:value,'write_report':lambda **v:{'written':True},'run_checks':lambda:{'passed':True}},
                        action_contract=contract,write_guard=guard())
                    self.assertEqual(result['status'],status);self.assertEqual(result['api_calls'],calls)
                    self.assertEqual(result['prompt_tokens'],calls*100)
                finally:a.close()

    def test_plan_keeps_repeat_clusters_and_same_ten_step_budget_for_every_arm(self):
        config=json.loads(Path('configs/fast_deepseek.json').read_text(encoding='utf-8'))
        episodes,manifest=plan(config)
        self.assertEqual(len(episodes),36);self.assertEqual(len({e['base_id'] for e in episodes}),12)
        self.assertEqual(manifest['repeats'],3);self.assertEqual(manifest['config']['max_steps'],10)
        self.assertEqual(manifest['config']['write_transform_arms'],config['arms'])
        for change in [{'max_steps':6},{'repeats':100},{'max_calls':1},{'write_transform_arms':['context_fast']}]:
            with self.assertRaises(ValueError):plan({**config,**change})


if __name__=='__main__':unittest.main()
