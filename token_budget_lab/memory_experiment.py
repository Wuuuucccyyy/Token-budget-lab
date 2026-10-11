"""Plugin on/off ablation. Offline runs report evidence only, never model accuracy."""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import cluster_ci, latency_stats, mean
from .datasets import digest
from .experiment import ROOT, code_hash, stable_hash
from .memory import HistoryArchive, MemoryPlugin, encode
from .memory_agent import DeepSeekChat, run_agent
from .metrics import score_answer


def prepare(config):
    if config['backend'] not in {'offline','deepseek'} or config['split'] not in {'dev','test'}:
        raise ValueError('invalid backend/split')
    if set(config['modes']) != {'full','compact','recover'} or len(config['modes']) != 3:
        raise ValueError('full/compact/recover ablation required')
    path=ROOT/config['data']
    sessions=[s for s in json.loads(path.read_text(encoding='utf-8'))['sessions'] if s['split']==config['split']]
    random.Random(config['seed']).shuffle(sessions)
    if config.get('session_limit'):
        sessions=sessions[:config['session_limit']]
    rows=[dict(q,session_id=s['id'],messages=s['messages']+[dict(role='user',content=q['query'])])
          for i in range(max(len(s['questions']) for s in sessions)) for s in sessions for q in s['questions'][i:i+1]]
    if config['backend']=='deepseek':
        for field in ('model','max_tokens','max_calls','max_steps'):
            if field not in config or not config[field]:
                raise ValueError(f'explicit {field} required')
    manifest=dict(schema_version='agent-memory-v1',config=config,data_sha256=digest(path),code_sha256=code_hash(),
                  question_ids=[r['id'] for r in rows],session_ids=[s['id'] for s in sessions])
    manifest['identity']=stable_hash(manifest)
    return rows,manifest


def summarize(records,manifest):
    config=manifest['config']
    rows=[r for r in records if r['event']=='task_result']
    groups=defaultdict(list)
    for row in rows:
        groups[row['mode']].append(row)
    steps=defaultdict(list)
    attempts=Counter()
    for r in records:
        if r['event']=='response':
            steps[r['mode']].append(r)
        elif r['event']=='start':
            attempts[r['mode']]+=1
    baseline={r['id']:r for r in groups['full']}
    offline=config['backend']=='offline'
    runs=[]
    for mode in config['modes']:
        rs=groups[mode]
        usages=[s['usage'] for s in steps[mode] if s.get('usage') is not None]
        pairs=[dict(document_id=r['session_id'],delta=r['exact_match']-baseline[r['id']]['exact_match'])
               for r in rs if not offline and r['id'] in baseline]
        runs.append(dict(mode=mode,questions=len(rs),expected_questions=len(manifest['question_ids']),
                         exact_match=mean([r['exact_match'] for r in rs]),f1=mean([r['f1'] for r in rs]),
                         agreement_with_full=mean([r['normalized_answer']==baseline[r['id']]['normalized_answer']
                             and r['status']=='ok' and baseline[r['id']]['status']=='ok' for r in rs if not offline and r['id'] in baseline]),
                         source_messages_retained=mean([r['source_messages_retained'] for r in rs]),
                         paired_quality_difference=cluster_ci(pairs,'delta',config['seed']),
                         statuses=dict(Counter(r['status'] for r in rs)),
                         initial_history_modes=dict(Counter(r['initial_history_mode'] for r in rs)),
                         initial_history_reasons=dict(Counter(r['initial_history_reason'] for r in rs)),
                         mean_initial_units=mean([r['initial_sent_units'] for r in rs]),
                         api_calls=attempts[mode],
                         recovery_tool_calls=sum(c.get('function',{}).get('name','').startswith('history_')
                             for s in steps[mode] for c in s.get('message',{}).get('tool_calls',[])),
                         prompt_tokens=None if offline else sum(u['prompt_tokens'] for u in usages),
                         completion_tokens=None if offline else sum(u['completion_tokens'] for u in usages),
                         unknown_usage_requests=0 if offline else attempts[mode]-len(usages),
                         latency=latency_stats(rs)))
    base=next(r for r in runs if r['mode']=='full')
    for r in runs:
        r['input_token_reduction']=1-r['prompt_tokens']/base['prompt_tokens'] if not offline and base['prompt_tokens'] else None
        r['total_token_reduction']=1-(r['prompt_tokens']+r['completion_tokens'])/(base['prompt_tokens']+base['completion_tokens']) if not offline and base['prompt_tokens'] else None
        r['quality_status']='proxy_only' if offline else 'pilot_only_not_certified'
    return dict(manifest=manifest,runs=runs)


def execute(config,output,transport=None):
    rows,manifest=prepare(config)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    log_path=output/'requests.jsonl'
    if log_path.exists() or (output/'manifest.json').exists():
        raise ValueError('existing experiment; use a fresh directory (no automatic paid replay)')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=HistoryArchive(output/'history.sqlite3')
    plugin=MemoryPlugin(archive,budget=config['budget_characters'],recent_turns=config['recent_turns'])
    records,attempts=[],0
    rng=random.Random(config['seed'])
    if config['backend']=='deepseek' and transport is None:
        transport=DeepSeekChat(config['model'],config['max_tokens'],config.get('timeout',120))
    try:
        with log_path.open('x',encoding='utf-8') as journal:
            def append(record):
                journal.write(json.dumps(record,ensure_ascii=False)+'\n')
                journal.flush()
                os.fsync(journal.fileno())
                records.append(record)
            for row in rows:
                modes=config['modes'].copy()
                rng.shuffle(modes)
                for mode in modes:
                    session=f"{row['id']}:{mode}"
                    r=dict(event='task_result',id=row['id'],session_id=row['session_id'],kind=row['kind'],mode=mode,
                           exact_match=None,f1=None,normalized_answer=None,
                           status='diagnostic',api_calls=0,recovery_tool_calls=0,prompt_tokens=0,completion_tokens=0,
                           unknown_usage_requests=0)
                    if config['backend']=='deepseek':
                        first=[]
                        def before(step,p):
                            nonlocal attempts
                            if attempts>=config['max_calls']:
                                raise RuntimeError('paid call limit reached')
                            append(dict(event='start',id=row['id'],mode=mode,step=step,
                                sent_messages_sha256=hashlib.sha256(encode(p.messages).encode()).hexdigest()))
                            attempts+=1
                            if step==0:
                                first.append(p)
                        def after(step):
                            append(dict(event='response',id=row['id'],mode=mode,**step))
                        result=run_agent(row['messages'],transport,plugin,'benchmark',session,mode,
                                         config['max_steps'],before,after)
                        prepared=first[0]
                        answer=''
                        if result['status']=='ok':
                            try:
                                parsed=json.loads(result['answer'])
                                if not isinstance(parsed,dict) or set(parsed)!={'answer'} or not isinstance(parsed['answer'],str):
                                    raise ValueError('invalid final answer')
                                answer=parsed['answer']
                            except (ValueError,TypeError):
                                result['status']='invalid_json'
                        r.update({k:result[k] for k in ('status','api_calls','recovery_tool_calls','prompt_tokens',
                                  'completion_tokens','unknown_usage_requests','end_to_end_seconds')})
                        from .metrics import normalize
                        r.update(**score_answer(answer,row['answers']),normalized_answer=normalize(answer),answer=answer)
                    else:
                        prepared=plugin.before_model('benchmark',session,row['messages'],mode)
                        r['end_to_end_seconds']=prepared.preparation_seconds
                    # Labels enter only after preparation/generation; never the plugin.
                    sources=row['evidence_positions']
                    retained=(all(row['messages'][p] in prepared.messages for p in sources) if sources is not None else None)
                    r.update(initial_history_mode=prepared.mode,initial_history_reason=prepared.reason,
                             initial_sent_units=prepared.sent_units,baseline_units=prepared.baseline_units,
                             source_messages_retained=retained)
                    append(r)
                    if r['status']=='error':
                        raise RuntimeError('API failure logged; no automatic retry')
    finally:
        archive.close()
        summary=summarize(records,manifest)
        summary['api_attempts']=attempts
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def export(output,destination):
    output,destination=Path(output),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','summary.json'):
        (destination/name).write_bytes((output/name).read_bytes())
    fields=('id','session_id','kind','mode','status','exact_match','f1','initial_history_mode',
            'initial_history_reason','initial_sent_units','baseline_units','source_messages_retained',
            'api_calls','recovery_tool_calls','prompt_tokens','completion_tokens','unknown_usage_requests','end_to_end_seconds')
    with (destination/'per_question.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for line in (output/'requests.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['event']=='task_result':
                writer.writerow({f:row[f] for f in fields})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.plan:
        rows,manifest=prepare(config)
        print(json.dumps(dict(questions=len(rows),independent_sessions=len(manifest['session_ids']),
                         minimum_calls=0 if config['backend']=='offline' else len(rows)*3,
                         maximum_calls=config.get('max_calls'),max_steps=config.get('max_steps')),indent=2))
        return
    execute(config,args.output)
    if args.export:
        export(args.output,args.export)
    print(f'Experiment written: {args.output.resolve()}')


if __name__=='__main__':
    main()
