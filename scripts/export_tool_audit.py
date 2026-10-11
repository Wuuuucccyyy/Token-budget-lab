"""Publish numeric request accounting and file-tool audit for synthetic episodes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def write_csv(path,fields,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.journal.read_bytes()
    records=[json.loads(l) for l in raw.decode('utf-8').splitlines()]
    args.output.mkdir(parents=True,exist_ok=True)
    requests=[];actions=[];states=[]
    for r in records:
        identity={k:r[k] for k in ('episode_id','checkpoint','mode')}
        if r['event']=='response':
            u=r.get('usage') or {}
            calls=[c['function']['name'] for c in r.get('message',{}).get('tool_calls',[])]
            requests.append(dict(**identity,step=r['step'],status=r['status'],history_mode=r['history_mode'],
                history_reason=r['history_reason'],prompt_tokens=u.get('prompt_tokens'),completion_tokens=u.get('completion_tokens'),
                prompt_cache_hit_tokens=u.get('prompt_cache_hit_tokens'),prompt_cache_miss_tokens=u.get('prompt_cache_miss_tokens'),
                context_metadata=json.dumps(r.get('context_metadata',{}),sort_keys=True),
                base_id=r.get('base_id',r['episode_id']),context_length=r.get('context_length','legacy'),
                baseline_characters=r.get('baseline_units'),sent_characters=r.get('sent_units'),
                recovery_tools_enabled=r.get('recovery_tools_enabled'),
                recovery_rounds=r.get('recovery_rounds'),full_history_locked=r.get('full_history_locked'),
                recovery_guard_reason=r.get('recovery_guard_reason'),
                suppressed_snapshot_reads=json.dumps(r.get('suppressed_snapshot_reads',[])),
                evidence_plan=json.dumps(r.get('evidence_plan',{}),ensure_ascii=False,sort_keys=True),
                api_latency_seconds=r.get('api_latency_seconds'),end_to_end_seconds=r['end_to_end_seconds'],
                restored_turn_ids=json.dumps(r['restored_turn_ids']),tool_names=json.dumps(calls)))
        elif r['event']=='business_tool':
            actions.append(dict(**identity,tool=r['tool'],success=r['success'],error_type=r.get('error_type'),
                arguments=json.dumps(r['arguments'],ensure_ascii=False,sort_keys=True),
                result=json.dumps(r.get('result'),ensure_ascii=False,sort_keys=True)))
        elif r['event']=='task_result':
            states.append(dict(**identity,task_success=r['task_success'],observed_state=r['observed_state']))
    if requests:write_csv(args.output/'per_request.csv',list(requests[0]),requests)
    if actions:write_csv(args.output/'business_actions.csv',list(actions[0]),actions)
    (args.output/'artifact_states.json').write_text(json.dumps(states,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest=dict(journal_sha256=hashlib.sha256(raw).hexdigest(),responses=len(requests),
        business_actions=len(actions),checkpoint_states=len(states),contains_api_credentials=False,
        scope='original synthetic tasks only; no real user files')
    (args.output/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
