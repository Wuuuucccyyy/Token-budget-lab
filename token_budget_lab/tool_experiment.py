"""Continuous file-tool episodes with a native baseline and fully metered recovery."""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

from .analysis import cluster_ci, latency_stats, mean
from .datasets import digest
from .experiment import ROOT, code_hash, stable_hash
from .memory import HistoryArchive, MemoryPlugin
from .memory_agent import DeepSeekChat, run_agent
from .scripted_policy import ScriptedPolicy
from .tool_tasks import TOOLS, ToolSandbox, assess


def prepare(config):
    if config['backend'] not in {'offline','deepseek'} or config['split'] not in {'dev','test'}:
        raise ValueError('invalid backend/split')
    if not config['modes'] or len(set(config['modes']))!=len(config['modes']) or 'full' not in config['modes'] or not set(config['modes'])<={'full','compact','recover'}:
        raise ValueError('unique modes and native full baseline required')
    if config['max_steps']<1 or config['budget_characters']<1:
        raise ValueError('invalid step/budget')
    path=ROOT/config['data']
    bundle=json.loads(path.read_text(encoding='utf-8'))
    groups={kind:[] for kind in ('knowledge_update','temporal','abstention','tool_execution')}
    for e in bundle['episodes']:
        if e['split']==config['split']: groups[e['kind']].append(e)
    rng=random.Random(config['seed'])
    episodes=[]
    for group in groups.values():
        rng.shuffle(group)
        episodes+=group[:config.get('per_kind_limit',len(group))]
    rng.shuffle(episodes)
    if not episodes: raise ValueError('no episodes')
    if config['backend']=='deepseek' and (config.get('max_calls',0)<1 or not config.get('model') or config.get('max_tokens',0)<1):
        raise ValueError('explicit paid-call/model/output limits required')
    manifest=dict(schema_version='continuous-tools-v1',config=config,data_sha256=digest(path),code_sha256=code_hash(),
        episode_ids=[e['id'] for e in episodes],checkpoints=sum(len(e['checkpoints']) for e in episodes),
        dataset_source=bundle['source'],baseline='native full history without memory tools/archive',
        setup='scripted actual file reads; no setup LLM calls; checkpoints are continuous agent execution',
        protocol='chat',thinking='disabled',json_output=True)
    manifest['identity']=stable_hash(manifest)
    return episodes,manifest


def summarize(records,manifest):
    offline=manifest['config']['backend']=='offline'
    tasks=[r for r in records if r['event']=='task_result']
    baseline={(r['episode_id'],r['checkpoint']):r for r in tasks if r['mode']=='full'}
    runs=[]
    for mode in manifest['config']['modes']:
        rows=[r for r in tasks if r['mode']==mode]
        responses=[r for r in records if r['event']=='response' and r['mode']==mode]
        starts=[r for r in records if r['event']=='start' and r['mode']==mode]
        usage=[r['usage'] for r in responses if r.get('usage') is not None]
        pairs=[dict(document_id=r['episode_id'],delta=int(r['task_success'])-int(baseline[(r['episode_id'],r['checkpoint'])]['task_success'])) for r in rows if (r['episode_id'],r['checkpoint']) in baseline]
        complete_episodes=[]
        for episode_id in manifest['episode_ids']:
            rs=[r for r in rows if r['episode_id']==episode_id]
            if len(rs)==3: complete_episodes.append(all(r['task_success'] for r in rs))
        agree=[r['format_ok'] and baseline[(r['episode_id'],r['checkpoint'])]['format_ok'] and r['answer']==baseline[(r['episode_id'],r['checkpoint'])]['answer'] for r in rows if (r['episode_id'],r['checkpoint']) in baseline]
        run=dict(mode=mode,checkpoints=len(rows),expected_checkpoints=manifest['checkpoints'],
            independent_episodes=len(complete_episodes),
            task_success_rate=None if offline else mean([r['task_success'] for r in rows]),
            episode_success_rate=None if offline else mean(complete_episodes),
            answer_accuracy=None if offline else mean([r['answer_correct'] for r in rows]),
            format_compliance=None if offline else mean([r['format_ok'] for r in rows]),
            artifact_state_accuracy=None if offline else mean([r['state_correct'] for r in rows]),
            required_tool_completion=None if offline else mean([r['required_tools_completed'] for r in rows]),
            constraint_violations=sum(r['constraint_violation'] for r in rows),
            agreement_with_full=None if offline else mean(agree),
            paired_quality_difference=None if offline else cluster_ci(pairs,'delta',manifest['config']['seed']),
            diagnostic_driver_success=sum(r['task_success'] for r in rows) if offline else None,
            status_counts=dict(Counter(r['status'] for r in rows)),
            initial_history_modes=dict(Counter(r['initial_history_mode'] for r in rows)),
            mean_initial_characters=mean([r['initial_sent_units'] for r in rows]),
            source_messages_retained=mean([r['source_messages_retained'] for r in rows]),
            model_calls=0 if offline else len(starts),scripted_policy_steps=len(starts) if offline else 0,
            business_tool_calls=sum(r['business_tool_calls'] for r in rows),
            recovery_tool_calls=sum(c['function']['name'].startswith('history_') for r in responses for c in r.get('message',{}).get('tool_calls',[])),
            prompt_tokens=None if offline else sum(u['prompt_tokens'] for u in usage),
            completion_tokens=None if offline else sum(u['completion_tokens'] for u in usage),
            unknown_usage_requests=0 if offline else len(starts)-len(usage),latency=latency_stats(rows),
            quality_status='scripted_driver_only_not_llm_accuracy' if offline else 'pilot_only_not_certified')
        runs.append(run)
    full=next(r for r in runs if r['mode']=='full')
    for r in runs:
        base=(full['prompt_tokens'] or 0)+(full['completion_tokens'] or 0)
        r['total_token_reduction']=None if offline or not base else 1-(r['prompt_tokens']+r['completion_tokens'])/base
    return dict(manifest=manifest,runs=runs,api_attempts=sum(r['model_calls'] for r in runs))


def execute(config,output,transport=None):
    episodes,manifest=prepare(config)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() or (output/'requests.jsonl').exists():
        raise ValueError('fresh output required; no automatic paid replay')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    records=[]
    attempts=0
    rng=random.Random(config['seed'])
    archive=HistoryArchive(output/'history.sqlite3')
    plugin=MemoryPlugin(archive,budget=config['budget_characters'],recent_turns=config['recent_turns'])
    if config['backend']=='deepseek' and transport is None:
        transport=DeepSeekChat(config['model'],config['max_tokens'],config.get('timeout',120),json_output=True)
    try:
        with (output/'requests.jsonl').open('x',encoding='utf-8') as log:
            def append(record):
                log.write(json.dumps(record,ensure_ascii=False)+'\n'); log.flush(); os.fsync(log.fileno())
                records.append(record)
            for episode in episodes:
                modes=config['modes'].copy(); rng.shuffle(modes)
                for mode in modes:
                    phase=[-1]
                    def on_action(action):
                        if phase[0]>=0: append(dict(event='business_tool',episode_id=episode['id'],mode=mode,checkpoint=phase[0],**action))
                    sandbox=ToolSandbox(output/'sandboxes'/episode['id']/mode,episode,on_action)
                    history,sources=sandbox.seed_history(episode)
                    client=transport if config['backend']=='deepseek' else ScriptedPolicy()
                    for number,checkpoint in enumerate(episode['checkpoints']):
                        phase[0]=number
                        history.append(dict(role='user',content=checkpoint['query']))
                        before_hashes=sandbox.file_hashes()
                        trace_start=len(sandbox.trace)
                        prepared=[]
                        def before(step,p):
                            nonlocal attempts
                            if config['backend']=='deepseek' and attempts>=config['max_calls']:
                                raise RuntimeError('paid call limit reached')
                            append(dict(event='start',episode_id=episode['id'],mode=mode,checkpoint=number,step=step,
                                sent_messages_sha256=stable_hash(p.messages)))
                            if config['backend']=='deepseek': attempts+=1
                            if step==0: prepared.append(p)
                        def after(record):
                            append(dict(event='response',episode_id=episode['id'],mode=mode,checkpoint=number,**record))
                        result=run_agent(history,client,None if mode=='full' else plugin,'tools',episode['id']+':'+mode,
                            mode=mode,max_steps=config['max_steps'],before_request=before,after_response=after,
                            extra_tools=TOOLS,handlers=sandbox.handlers(),memory_tools=mode=='recover')
                        # Canonical history persists across checkpoints, including mistakes.
                        original=history
                        history=result['messages']
                        score=assess(checkpoint,result,sandbox,before_hashes,trace_start,episode['initial_config'])
                        relevant=[]
                        snapshots=[s for s in sources if s['tool']=='read_config']
                        if episode['kind']=='knowledge_update' and number<2:
                            relevant=[snapshots[-1 if number==0 else 0]['position']]
                        elif episode['kind']=='temporal' and number<2:
                            relevant=[s['position'] for s in snapshots]
                        elif episode['kind']=='tool_execution' and number==0:
                            relevant=[s['position'] for s in sources if s['tool']=='read_orders']
                        retained=all(original[p] in prepared[0].messages for p in relevant) if relevant else None
                        row=dict(event='task_result',episode_id=episode['id'],kind=episode['kind'],checkpoint=number,mode=mode,
                            status=result['status'],**score,initial_history_mode=prepared[0].mode,
                            initial_history_reason=prepared[0].reason,initial_sent_units=prepared[0].sent_units,
                            source_messages_retained=retained,end_to_end_seconds=result['end_to_end_seconds'])
                        append(row)
                        if result['status']=='error': raise RuntimeError('API failure logged; no retry')
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
    rows=[json.loads(l) for l in (output/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    fields=['episode_id','kind','checkpoint','mode','status','format_ok','answer_correct','state_correct',
            'required_tools_completed','constraint_violation','task_success','business_tool_calls','initial_history_mode',
            'initial_history_reason','initial_sent_units','source_messages_retained','end_to_end_seconds']
    with (destination/'per_checkpoint.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:
            if r['event']=='task_result': writer.writerow({f:r[f] for f in fields})
    with (destination/'final_answers.csv').open('w',encoding='utf-8',newline='') as handle:
        fields=['episode_id','checkpoint','mode','raw_answer']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        last={}
        for r in rows:
            key=r.get('episode_id'),r.get('checkpoint'),r.get('mode')
            if r['event']=='response': last[key]=r.get('message',{}).get('content') or ''
            if r['event']=='task_result': writer.writerow(dict(episode_id=key[0],checkpoint=key[1],mode=key[2],raw_answer=last.get(key,'')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--export',type=Path)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.plan:
        episodes,manifest=prepare(config)
        print(json.dumps(dict(episodes=len(episodes),checkpoints=manifest['checkpoints'],modes=config['modes'],
            maximum_model_calls=0 if config['backend']=='offline' else config['max_calls'],
            max_steps_per_checkpoint=config['max_steps']),indent=2));return
    execute(config,args.output)
    if args.export: export(args.output,args.export)
    print(f'Written: {args.output.resolve()}')


if __name__=='__main__': main()
