"""Inspect the v0.6 context view; live execution requires an explicit switch."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from token_budget_lab.context_plugin import ContextPlugin
from token_budget_lab.context_tasks import build_context_tasks
from token_budget_lab.memory import HistoryArchive
from token_budget_lab.memory_agent import DeepSeekChat, run_agent
from token_budget_lab.provenance import HistoricalWriteGuard
from token_budget_lab.tool_tasks import ToolSandbox, TOOLS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',action='store_true');p.add_argument('--mode',choices=['full','context','guarded'],default='context')
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        episode=next(e for e in build_context_tasks()['episodes'] if e['kind']=='temporal')
        sandbox=ToolSandbox(Path(tmp)/'sandbox',episode)
        messages,_=sandbox.seed_history(episode)
        messages.append(dict(role='user',content=episode['checkpoints'][1]['query']))
        archive=HistoryArchive(Path(tmp)/'history.db')
        try:
            plugin=ContextPlugin(archive)
            if not args.live:
                view=plugin.before_model('demo','session',messages)
                print(json.dumps(dict(mode=view.mode,baseline_characters=view.baseline_units,
                    sent_characters=view.sent_units,metadata=view.context_metadata,
                    provenance_contract=HistoricalWriteGuard(messages).notice()),ensure_ascii=True,indent=2))
                return
            result=run_agent(messages,DeepSeekChat(json_output=True),None if args.mode=='full' else plugin,
                'demo','session',mode='full' if args.mode=='full' else 'recover',memory_tools=args.mode!='full',
                max_steps=6,extra_tools=TOOLS,handlers=sandbox.handlers(),
                write_guard=HistoricalWriteGuard(messages) if args.mode=='guarded' else None)
            print(json.dumps(dict(status=result['status'],answer=result['answer'],api_calls=result['api_calls'],
                prompt_tokens=result['prompt_tokens'],completion_tokens=result['completion_tokens']),ensure_ascii=True,indent=2))
        finally:archive.close()


if __name__=='__main__':main()
