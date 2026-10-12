import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.action_experiment import plan
from token_budget_lab.action_tasks import execution_contract, update_current_files
from token_budget_lab.evidence_locator import EvidenceLocator
from token_budget_lab.locator_tasks import build_locator_tasks
from token_budget_lab.memory import encode
from token_budget_lab.tool_tasks import ToolSandbox


class LocatorTaskTests(unittest.TestCase):
    def test_reordered_sparse_long_and_duplicate_cases_have_actual_original_locations(self):
        tasks=build_locator_tasks()
        self.assertEqual(tasks,json.loads(Path('data/locator_tasks.json').read_text(encoding='utf-8')))
        expectations={'locator-reordered':('scan_d','scan_d-43','/events/5',1),
                      'locator-sparse':('scan_c','scan_c-901','/events/7',1),
                      'locator-long-object':('scan_a','scan_a-37','/events/37',1),
                      'locator-duplicate':('scan_b','scan_b-17',None,2),
                      'locator-missing':('scan_b','scan_b-404',None,0)}
        with tempfile.TemporaryDirectory() as tmp:
            for e in tasks['episodes']:
                feature=e.get('locator_feature')
                identity='locator-'+feature if feature else ''
                if identity not in expectations:
                    continue
                sandbox=ToolSandbox(Path(tmp)/e['id'],e)
                sandbox.seed_history(e)
                update_current_files(sandbox,e['live_updates'])
                name,identifier,pointer,count=expectations[identity]
                index=EvidenceLocator({'s':encode(sandbox.read_log(name))})
                matches=index.find('s','event',identifier,['status_code'])
                self.assertEqual(matches['match_count'],count)
                if pointer:
                    self.assertEqual(matches['matches'][0]['pointer'],pointer)
                original=execution_contract(e['request_actions']).notice()
                e['checkpoints'][0].update(answer='GOLD_MUST_NOT_ENTER',timeout_ms=-1,report={'total':'SECRET'})
                self.assertEqual(original,execution_contract(e['request_actions']).notice())
                self.assertNotIn('GOLD_MUST_NOT_ENTER',original)

    def test_conditional_obligations_and_identifiers_do_not_expose_answer_branch(self):
        tasks=build_locator_tasks()['episodes']
        conditional=[e for e in tasks if e.get('locator_feature') not in {None,'text-tail'}]
        requirements=[{**e['request_actions'],'required':[{**a,'arguments':{}} for a in e['request_actions']['required']]}
                      for e in conditional]
        self.assertTrue(all(x==requirements[0] for x in requirements))
        self.assertTrue(all(len(e['request_actions']['required'])==1 for e in conditional))
        self.assertTrue(all(e['id']==f'lookup-{i:03d}' and e['base_id']==e['id'] for i,e in enumerate(tasks,1)))

    def test_plan_uses_same_contract_and_original_step_cap_for_three_arms(self):
        config=json.loads(Path('configs/locator_deepseek.json').read_text(encoding='utf-8'))
        episodes,manifest=plan(config)
        self.assertEqual(len(episodes),12)
        self.assertEqual(manifest['config']['max_steps'],6)
        self.assertEqual(manifest['config']['max_calls'],216)
        self.assertTrue(manifest['config']['execution_contract_all_arms'])
        bad=copy.deepcopy(config)
        bad['execution_contract_all_arms']=False
        with self.assertRaises(ValueError):
            plan(bad)
