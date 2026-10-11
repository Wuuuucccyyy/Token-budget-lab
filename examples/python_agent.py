"""Inspect or execute a current-file task with host execution obligations."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_budget_lab.action_tasks import build_action_tasks, execution_contract, update_current_files
from token_budget_lab.context_plugin import ContextPlugin
from token_budget_lab.memory import HistoryArchive
from token_budget_lab.memory_agent import DeepSeekChat, run_agent
from token_budget_lab.tool_tasks import ToolSandbox, TOOLS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live', action='store_true')
    p.add_argument('--mode', choices=['full', 'context'], default='context')
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        episode = next(e for e in build_action_tasks()['episodes'] if e['id']=='actions-changed-log-00')
        sandbox = ToolSandbox(Path(tmp)/'files', episode)
        messages, _ = sandbox.seed_history(episode)
        update_current_files(sandbox, episode['live_updates'])
        messages.append(dict(role='user', content=episode['checkpoints'][0]['query']))
        contract = execution_contract(episode['request_actions'])
        archive = HistoryArchive(Path(tmp)/'history.sqlite3')
        try:
            plugin = ContextPlugin(archive)
            if not args.live:
                prepared = plugin.before_model('demo', 'session', messages)
                print(json.dumps(dict(contract=contract.definition(), mode=prepared.mode,
                    baseline_characters=prepared.baseline_units, sent_characters=prepared.sent_units), ensure_ascii=True, indent=2))
                return
            result = run_agent(messages, DeepSeekChat(json_output=True), plugin if args.mode=='context' else None,
                'demo', 'session', mode='recover' if args.mode=='context' else 'full', memory_tools=args.mode=='context',
                max_steps=6, max_recovery_rounds=3, extra_tools=TOOLS, handlers=sandbox.handlers(), action_contract=contract)
            print(json.dumps({k:result[k] for k in ('status', 'answer', 'action_contract', 'api_calls', 'prompt_tokens', 'completion_tokens')},
                             ensure_ascii=True, indent=2))
        finally:
            archive.close()


if __name__=='__main__':
    main()
