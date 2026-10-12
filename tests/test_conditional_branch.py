import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from helpers import Script
from token_budget_lab import ContextPlugin, FreshWriteGuard, HistoryArchive, WriteTransform, run_agent
from token_budget_lab.action_experiment import execute
from token_budget_lab.experiment import ROOT
from token_budget_lab.memory import encode, validate_messages
from token_budget_lab.tool_tasks import TOOLS
from token_budget_lab.transform_tasks import guard_from_query
from token_budget_lab.write_transform import BranchActionRejected


def guard():
    return FreshWriteGuard([WriteTransform('read_log',{'name':'scan_b'},'write_report','total',
        {'kind':'unique_event','array':'/events','match_key':'event','match_value':'scan_b-17','field':'status_code'},
        fixed_arguments={'currency':'USD'})],conditional_read_only=True)


def observe(g,events):
    value={'events':events,'detail':'background '*500}
    message={'role':'tool','tool_call_id':'read','content':encode(value)}
    g.before_invocation('read_log',{'name':'scan_b'})
    g.observe('read_log',{'name':'scan_b'},value,message,3,True)


class DeliberateBadWrite:
    """Test-only driver ignoring the notice; never an accuracy measurement."""
    last={}
    calls=0
    def __call__(self,messages,tools):
        self.calls+=1
        boundary=max(i for i,m in enumerate(messages) if m['role']=='user')
        returns=[m for m in messages[boundary:] if m['role']=='tool']
        name='read_log' if not returns else 'write_report' if len(returns)==1 else None
        args={'name':'scan_b'} if name=='read_log' else {'total':'UNKNOWN','currency':'USD'}
        message={'role':'assistant','content':None if name else '{"answer":"UNKNOWN"}'}
        if name:message['tool_calls']=[{'id':f'bad-{self.calls}','type':'function','function':{'name':name,'arguments':encode(args)}}]
        return {'message':message,'usage':{'prompt_tokens':100,'completion_tokens':10,'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':100}}


class ConditionalBranchTests(unittest.TestCase):
    def test_initial_state_is_not_a_precomputed_answer_branch(self):
        g=guard();state=g.branch_state()
        self.assertEqual(state,{'selection_status':'awaiting_read','actions_authorized':False})
        self.assertNotIn('source_id',state);self.assertNotIn('refusal_answer',state)

    def test_actual_missing_and_ambiguous_reads_authorize_only_read_only_branch(self):
        for events,status in [([], 'missing'),([{'event':'scan_b-17','status_code':206}]*2,'ambiguous')]:
            g=guard();observe(g,events);state=g.branch_state()
            self.assertEqual(state['selection_status'],status);self.assertFalse(state['actions_authorized'])
            self.assertEqual(state['refusal_answer'],'UNKNOWN')
            for tool in ('write_config','write_report','run_checks'):
                with self.assertRaises(BranchActionRejected):g.validate(tool,{'total':'UNKNOWN','currency':'USD'})

    def test_unique_original_still_allows_actual_write_and_checks(self):
        g=guard();observe(g,[{'event':'scan_b-17','status_code':206}])
        self.assertTrue(g.branch_state()['actions_authorized'])
        args={'total':'206.00','currency':'USD'}
        self.assertEqual(g.validate('write_report',args),args);self.assertEqual(g.validate('run_checks',{}),{})

    def test_invalid_format_is_unresolved_not_false_missing(self):
        g=guard();observe(g,[{'event':'scan_b-17'}]);state=g.branch_state()
        self.assertEqual(state['selection_status'],'unresolved');self.assertNotIn('refusal_answer',state)

    def test_failed_new_read_invalidates_prior_unique_permission(self):
        g=guard();observe(g,[{'event':'scan_b-17','status_code':206}])
        g.before_invocation('read_log',{'name':'scan_b'})
        self.assertEqual(g.branch_state()['selection_status'],'awaiting_read')
        with self.assertRaises(BranchActionRejected):g.validate('write_report',{'total':'206.00','currency':'USD'})

    def test_full_and_plugin_receive_same_read_only_notice_and_keep_raw_history(self):
        first_notice=None
        for compressed in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                archive=HistoryArchive(Path(tmp)/'a.db')
                try:
                    rows=[{'role':'system','content':'Use current tools'}, {'role':'user','content':'If missing, answer UNKNOWN and do not write.'}]
                    body={'events':[],'detail':'background '*500}
                    script=Script([[('read_log',{'name':'scan_b'})],'{"answer":"UNKNOWN"}'])
                    result=run_agent(rows,script,ContextPlugin(archive) if compressed else None,'t','s',
                        mode='recover' if compressed else 'full',memory_tools=compressed,write_guard=guard(),extra_tools=TOOLS,
                        handlers={'read_log':lambda name:body,'write_config':lambda **v:None,'write_report':lambda **v:None,'run_checks':lambda:None})
                    notice=next(m['content'] for m in script.requests[1][0] if isinstance(m.get('content'),str) and m['content'].startswith('[Current-run conditional authorization snapshot]'))
                    if first_notice is None:first_notice=notice
                    else:self.assertEqual(notice,first_notice)
                    self.assertEqual(json.loads(result['messages'][3]['content']),body)
                    self.assertFalse(any(isinstance(m.get('content'),str) and 'conditional authorization snapshot' in m['content'] for m in result['messages']))
                    self.assertEqual(script.requests[0][1],script.requests[1][1])
                    self.assertEqual(script.requests[1][0][:len(script.requests[0][0])],script.requests[0][0])
                finally:archive.close()

    def test_notice_after_batched_tool_group_and_later_read_preserves_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive=HistoryArchive(Path(tmp)/'a.db')
            try:
                results=iter([{'events':[{'event':'scan_b-17','status_code':206}]},{'events':[]}])
                script=Script([[('read_log',{'name':'scan_b'}),('read_report',{})],[('read_log',{'name':'scan_b'})],'{"answer":"UNKNOWN"}'])
                run_agent([{'role':'user','content':'Read conditionally'}],script,ContextPlugin(archive),'t','s',extra_tools=TOOLS,write_guard=guard(),
                    handlers={'read_log':lambda name:next(results),'read_report':lambda:{'report':None},'write_config':lambda **v:None,
                        'write_report':lambda **v:None,'run_checks':lambda:None})
                for messages,_ in script.requests:validate_messages(messages)
                self.assertEqual(script.requests[2][0][:len(script.requests[1][0])],script.requests[1][0])
                notices=[m['content'] for m in script.requests[2][0] if isinstance(m.get('content'),str) and m['content'].startswith('[Current-run conditional authorization snapshot]')]
                self.assertEqual(len(notices),2);self.assertIn('"selection_status":"missing"',notices[-1])
            finally:archive.close()

    def test_host_adapter_uses_conditional_user_wording_not_expected_labels(self):
        fixtures=json.loads(Path('data/locator_tasks.json').read_text(encoding='utf-8'))['episodes']
        flags=[]
        for e in fixtures:
            g=guard_from_query(e['checkpoints'][0]['query']);flags.append(g.conditional_read_only)
            altered=copy.deepcopy(e);altered['checkpoints'][0].update(answer='FAKE',no_writes=not e['checkpoints'][0].get('no_writes',False))
            self.assertEqual(g.notice(),guard_from_query(altered['checkpoints'][0]['query']).notice())
        self.assertEqual(flags,[False]*6+[True]*5+[False])

    def test_numeric_tool_schema_does_not_offer_unknown_as_an_amount(self):
        t=next(t for t in TOOLS if t['function']['name']=='write_report')['function']
        pattern=t['parameters']['properties']['total']['pattern']
        self.assertIsNone(re.fullmatch(pattern,'UNKNOWN'));self.assertIsNotNone(re.fullmatch(pattern,'206.00'))

    def test_forbidden_attempt_still_fails_strict_scoring_even_if_host_blocks_it(self):
        config=json.loads(Path('configs/fast_deepseek.json').read_text(encoding='utf-8'))
        bundle=json.loads(Path('data/locator_tasks.json').read_text(encoding='utf-8'))
        missing=next(e for e in bundle['episodes'] if e['id']=='lookup-011')
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=Path(tmp)/'tasks.json';path.write_text(json.dumps({**bundle,'episodes':[missing]}),encoding='utf-8')
            config.update(data=path.relative_to(ROOT).as_posix(),repeats=1,max_calls=30)
            execute(config,Path(tmp)/'run',DeliberateBadWrite())
            rows=[json.loads(s) for s in (Path(tmp)/'run/requests.jsonl').read_text(encoding='utf-8').splitlines()]
            tasks=[r for r in rows if r['event']=='task_result']
            self.assertEqual(len(tasks),3)
            self.assertTrue(all(t['constraint_violation'] and not t['task_success'] and t['state_correct'] and t['model_write_attempts']==1 for t in tasks))
            self.assertFalse(any(r['event']=='business_tool' and r['tool']=='write_report' for r in rows))


if __name__=='__main__':unittest.main()
