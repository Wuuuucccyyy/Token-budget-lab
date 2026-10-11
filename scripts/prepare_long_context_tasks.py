"""Rebuild the original short/medium/long matched fixtures without API calls."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from token_budget_lab.long_context_tasks import build_long_tasks

if __name__ == '__main__':
    data = build_long_tasks()
    path = ROOT/'data/agent_long_context_tasks.json'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Written {len(data["episodes"])} episodes, 216 checkpoints; no API calls.')
