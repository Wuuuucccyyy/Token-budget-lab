import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from token_budget_lab.action_experiment import execute
from token_budget_lab.action_tasks import build_action_tasks, execution_contract, update_current_files
from token_budget_lab.memory import encode
from token_budget_lab.tool_tasks import ToolSandbox


class CurrentFileDriver:
    """Test-only scripted protocol driver; no accuracy measurement."""
    last = {}
    calls = 0

    def __call__(self, messages, tools):
        self.calls += 1
        index = max(i for i, m in enumerate(messages) if m['role']=='user')
        query = messages[index]['content']
        rows = [json.loads(m['content']) for m in messages[index:] if m['role']=='tool']
        name, args = None, {}
        if 'read_log' in query:
            log, event = re.search(r'event="(scan_[abcd])-(\d+)"', query).groups()
            event = int(event)
            if not rows:
                name, args = 'read_log', dict(name=log)
            elif not any('events' in r or 'text' in r for r in rows):
                handle = next(r for r in rows if r.get('archived'))
                name, args = 'context_read', dict(source_id=handle['source_id'], pointer=f'/events/{event}/status_code')
            else:
                raw = next(r for r in rows if 'events' in r or 'text' in r)
                value = json.loads(raw['text']) if 'text' in raw else raw['events'][event]['status_code']
                value = f'{value:.2f}'
                if not any(r.get('written') for r in rows):
                    name, args = 'write_report', dict(total=value, currency='USD')
                elif not any('passed' in r for r in rows):
                    name = 'run_checks'
        else:
            if not rows:
                name = 'read_config'
            else:
                value = str(next(r['config']['timeout_ms'] for r in rows if 'config' in r)+1)
                if not any(r.get('written') for r in rows):
                    name, args = 'write_config', dict(timeout_ms=int(value))
                elif not any('passed' in r for r in rows):
                    name = 'run_checks'
        message = dict(role='assistant', content=None if name else encode(dict(answer=value)))
        if name:
            message['tool_calls'] = [dict(id=f'test-{self.calls}', type='function',
                function=dict(name=name, arguments=encode(args)))]
        response = dict(message=message, usage=dict(prompt_tokens=100, completion_tokens=10,
                        prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=100), finish_reason='tool_calls' if name else 'stop')
        self.last = {k:v for k,v in response.items() if k!='message'}
        return response


class ActionExperimentTests(unittest.TestCase):
    def test_frozen_fixtures_current_changes_and_no_reference_in_contract(self):
        frozen = json.loads(Path('data/action_contract_tasks.json').read_text(encoding='utf-8'))
        self.assertEqual(frozen, build_action_tasks())
        with tempfile.TemporaryDirectory() as tmp:
            for n, e in enumerate(frozen['episodes']):
                a = ToolSandbox(Path(tmp)/f'a{n}', e)
                history, _ = a.seed_history(e)
                update_current_files(a, e['live_updates'])
                altered = copy.deepcopy(e)
                altered['checkpoints'][0].update(answer='FAKE GOLD', timeout_ms=-1, report={'total': 'WRONG'})
                b = ToolSandbox(Path(tmp)/f'b{n}', altered)
                other, _ = b.seed_history(altered)
                update_current_files(b, altered['live_updates'])
                self.assertEqual(history, other)
                self.assertEqual(a.file_hashes(), b.file_hashes())
                self.assertEqual(execution_contract(e['request_actions']).notice(),
                                 execution_contract(altered['request_actions']).notice())
                if e['live_updates']:
                    update = e['live_updates'][0]
                    original = [json.loads(m['content']) for m in history if m['role']=='tool']
                    if 'event' in update:
                        old = next(r for r in original if 'events' in r and r['events'][0]['event'].startswith(update['file'][:-5]))
                        self.assertNotEqual(old['events'][update['event']]['status_code'], update['fields']['status_code'])
                    else:
                        self.assertFalse(any(r.get('config', {}).get('timeout_ms')==update['fields']['timeout_ms'] for r in original))

    def test_four_arms_meter_complete_tasks_and_successful_execution_receipts(self):
        config = json.loads(Path('configs/action_contract_deepseek.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            summary = execute(config, Path(tmp)/'run', CurrentFileDriver())
            self.assertEqual([r['checkpoints'] for r in summary['runs']], [6]*4)
            self.assertEqual([r['episode_success_rate'] for r in summary['runs']], [1]*4)
            self.assertEqual([r['fresh_read_completed'] for r in summary['runs']], [6]*4)
            self.assertEqual(summary['unknown_usage_attempts'], 0)
            self.assertEqual(summary['total_api_attempts'], sum(r['model_calls'] for r in summary['runs']))
            rows = [json.loads(line) for line in (Path(tmp)/'run/requests.jsonl').read_text(encoding='utf-8').splitlines()]
            guarded = [r for r in rows if r['event']=='task_result' and r['mode'].endswith('_contract')]
            self.assertEqual(len(guarded), 12)
            self.assertTrue(all(r['execution_contract']['complete'] for r in guarded))
            self.assertTrue(all(len(r['execution_contract']['receipts'])==3 for r in guarded))
            with self.assertRaises(ValueError):
                execute(config, Path(tmp)/'run', CurrentFileDriver())


if __name__=='__main__':
    unittest.main()
