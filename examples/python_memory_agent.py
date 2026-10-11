"""Run the Python history hook offline, or explicitly request a paid DeepSeek run."""
import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from token_budget_lab.memory import HistoryArchive, MemoryPlugin
from token_budget_lab.memory_agent import DeepSeekChat, run_agent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Call DeepSeek with a locally configured DEEPSEEK_API_KEY')
    parser.add_argument('--mode',choices=['full','recover'],default='full')
    parser.add_argument('--model',default='deepseek-flash')
    parser.add_argument('--archive',type=Path,default=ROOT/'local_results/demo-memory.sqlite3')
    args=parser.parse_args()
    fixture=json.loads((ROOT/'data/agent_memory.json').read_text(encoding='utf-8'))['sessions'][0]
    question=fixture['questions'][0]
    canonical=fixture['messages']+[dict(role='user',content=question['query'])]
    archive=HistoryArchive(args.archive)
    plugin=MemoryPlugin(archive,budget=9000,recent_turns=2)
    session='demo-'+uuid.uuid4().hex
    try:
        if args.live:
            result=run_agent(canonical,DeepSeekChat(args.model,max_tokens=512),plugin,
                             tenant='local-demo',session=session,mode=args.mode,max_steps=3)
            print(json.dumps({key:result[key] for key in ('answer','status','api_calls',
                'prompt_tokens','completion_tokens','unknown_usage_requests','end_to_end_seconds')},ensure_ascii=False,indent=2))
            # Persist result['messages'] as the canonical state before appending the next user message.
            if result['status']!='ok':
                raise SystemExit(1)
        else:
            for mode in ('full','recover'):
                prepared=plugin.before_model('local-demo',session,canonical,mode=mode)
                print(json.dumps(dict(mode=mode,history_mode=prepared.mode,reason=prepared.reason,
                    unit=prepared.unit,original_units=prepared.baseline_units,sent_units=prepared.sent_units,
                    restored_turn_ids=prepared.restored_turn_ids),ensure_ascii=False))
            print('Offline history inspection only; no model answer or accuracy measured.')
    finally:
        archive.close()


if __name__=='__main__':
    main()
