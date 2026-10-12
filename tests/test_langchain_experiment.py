"""Strict task scoring and complete accounting on one actual LangChain graph."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import langchain
except ImportError:
    langchain=None

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipIf(langchain is None,'Install the langchain optional dependency')
class LangChainExperimentTests(unittest.TestCase):
    def test_host_policy_is_independent_of_expected_answer_and_state_labels(self):
        import copy
        from token_budget_lab.langchain_experiment import current_intent
        bundle=json.loads((ROOT/'data/locator_tasks.json').read_text(encoding='utf-8'))
        episode=next(e for e in bundle['episodes'] if e['checkpoints'][0].get('no_writes'))
        changed=copy.deepcopy(episode)
        changed['checkpoints'][0].update(answer='FAKE',no_writes=False,expected_config={'timeout_ms':999})
        first=current_intent(episode);second=current_intent(changed)
        self.assertEqual(first[0].definition(),second[0].definition())
        self.assertEqual(first[1].notice(),second[1].notice())
        self.assertEqual(first[2:],second[2:])

    def test_blocked_forbidden_write_still_fails_both_arms_and_all_calls_count(self):
        from langchain.agents import create_agent
        from token_budget_lab.langchain_experiment import execute
        from token_budget_lab.transform_tasks import guard_from_query
        from test_langchain_adapter import ScriptedTransport,call
        bundle=json.loads((ROOT/'data/locator_tasks.json').read_text(encoding='utf-8'))
        episode=next(e for e in bundle['episodes'] if e['checkpoints'][0].get('no_writes'))
        rule=guard_from_query(episode['checkpoints'][0]['query']).rules[0]
        sequence=[call(rule.read_tool,rule.read_arguments,'read'),
            call('write_report',dict(total='UNKNOWN',currency='USD'),'forbidden'),
            dict(role='assistant',content='{"answer":"UNKNOWN"}')]
        transport=ScriptedTransport(sequence+sequence)
        config=json.loads((ROOT/'configs/langchain_deepseek.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=Path(tmp)/'fixture.json'
            path.write_text(json.dumps(dict(source='single frozen case for scripted scoring test',episodes=[episode])),encoding='utf-8')
            config.update(data=path.relative_to(ROOT).as_posix(),repeats=1)
            with patch('token_budget_lab.langchain_experiment.create_agent',wraps=create_agent) as compiler:
                summary=execute(config,Path(tmp)/'run',transport)
            self.assertEqual(compiler.call_count,1)
            self.assertEqual(summary['actual_api_calls'],6)
            self.assertEqual(summary['unknown_usage_attempts'],0)
            for arm in summary['runs']:
                self.assertEqual(arm['task_success_rate'],0)
                self.assertEqual(arm['constraint_violations'],1)
                self.assertEqual(arm['fresh_read_completed'],1)
                self.assertEqual(arm['model_calls'],3)
                self.assertEqual(arm['prompt_tokens'],300)
            rows=[json.loads(line) for line in (Path(tmp)/'run/requests.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertFalse(any(r['event']=='business_tool' and r['tool']=='write_report' for r in rows))
            self.assertTrue(all(r['model_write_attempts']==1 for r in rows if r['event']=='task_result'))
