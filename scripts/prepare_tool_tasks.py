"""Rebuild original memory/tool fixtures; no model or external side effects."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from token_budget_lab.tool_tasks import build_tasks

if __name__=='__main__':
    target=ROOT/'data/agent_tool_tasks.json'
    target.write_text(json.dumps(build_tasks(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote 24 episodes, 72 continuous checkpoints; no API calls.')
