"""Scripted wiring verification only; no model quality or saving claim."""
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.context_experiment import execute
from token_budget_lab.context_tasks import build_context_tasks
from token_budget_lab.memory import encode
from token_budget_lab.scripted_policy import ScriptedPolicy
from token_budget_lab.tool_tasks import TOOLS


class FixtureDriver(ScriptedPolicy):
    def __call__(self,messages,tools):
        index=max(i for i,m in enumerate(messages) if m['role']=='user')
        query=messages[index]['content']
        if 'status_code' in query:
            self.calls+=1
            rows=[json.loads(m['content']) for m in messages[index:] if m['role']=='tool']
            target=int(re.search(r'scan_c-(\d+)',query).group(1))
            name,args=None,{}
            if not rows:name,args='read_log',dict(name='scan_c')
            elif not any('events' in r or 'text' in r for r in rows):
                handle=next(r for r in rows if r.get('archived'))
                name,args='context_read',dict(source_id=handle['source_id'],pointer=f'/events/{target}/status_code')
            else:
                raw=next((r for r in rows if 'events' in r or 'text' in r),None)
                value=json.loads(raw['text']) if 'text' in raw else raw['events'][target]['status_code']
                value=f'{value:.2f}'
                if not any(r.get('written') for r in rows):name,args='write_report',dict(total=value,currency='USD')
                elif not any('passed' in r for r in rows):name='run_checks'
                else:response=dict(message=dict(role='assistant',content=encode(dict(answer=value))),finish_reason='stop')
            if name:response=dict(message=dict(role='assistant',content=None,tool_calls=[dict(id=f'policy-{self.calls}',type='function',function=dict(name=name,arguments=encode(args)))]),finish_reason='tool_calls')
        else:
            response=super().__call__(messages,tools)
        for call in response['message'].get('tool_calls',[]):
            if call['function']['name']=='write_config':
                notices=[m['content'] for m in messages if isinstance(m.get('content'),str) and m['content'].startswith('[Write provenance contract]')]
                if notices and 'evidence=' in notices[-1]:
                    evidence=json.JSONDecoder().raw_decode(notices[-1].split('evidence=',1)[1])[0]
                    args=json.loads(call['function']['arguments']);args['evidence']=evidence
                    call['function']['arguments']=encode(args)
        # These synthetic counts exist ONLY inside this test's temporary output.
        response['usage']=dict(prompt_tokens=100,prompt_cache_hit_tokens=0,prompt_cache_miss_tokens=100,completion_tokens=10)
        self.last={k:v for k,v in response.items() if k!='message'}
        return response


class ExperimentWiringTests(unittest.TestCase):
    def test_four_arms_execute_files_guarded_writes_and_paid_probe_accounting(self):
        config=json.loads(Path('configs/context_plugin_deepseek.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            summary=execute(config,Path(tmp)/'run',FixtureDriver())
            self.assertEqual([r['checkpoints'] for r in summary['runs']],[24]*4)
            self.assertEqual([r['task_success_rate'] for r in summary['runs']],[1]*4)
            self.assertEqual(summary['probe_api_attempts'],32)
            self.assertEqual(summary['unknown_usage_attempts_including_probes'],0)
            self.assertGreaterEqual(next(r for r in summary['runs'] if r['mode']=='context')['recovery_tool_calls'],2)
            self.assertGreaterEqual(next(r for r in summary['runs'] if r['mode']=='context_guarded')['recovery_tool_calls'],2)
            self.assertEqual(summary['total_api_attempts_including_probes'],sum(r['model_calls'] for r in summary['runs'])+32)
            with self.assertRaises(ValueError):execute(config,Path(tmp)/'run',FixtureDriver())

    def test_dataset_is_frozen_and_facts_do_not_depend_on_gold(self):
        frozen=json.loads(Path('data/context_plugin_tasks.json').read_text(encoding='utf-8'))
        self.assertEqual(frozen,build_context_tasks())
        from token_budget_lab.tool_tasks import ToolSandbox
        e=copy.deepcopy(frozen['episodes'][0])
        with tempfile.TemporaryDirectory() as tmp:
            a=ToolSandbox(Path(tmp)/'a',e);ha,_=a.seed_history(e)
            for cp in e['checkpoints']:cp.update(answer='ARBITRARY',timeout_ms=-123)
            b=ToolSandbox(Path(tmp)/'b',e);hb,_=b.seed_history(e)
            self.assertEqual(ha,hb)
