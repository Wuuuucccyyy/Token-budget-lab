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
        if e['split']==config['split'] and (not config.get('context_lengths') or e.get('context_length','legacy') in config['context_lengths']):
            groups[e['kind']].append(e)
    rng=random.Random(config['seed'])
    episodes=[]
    for group in groups.values():
        # Length variants of one scenario stay matched and count as one cluster.
        base_ids=sorted({e.get('base_id',e['id']) for e in group})
        rng.shuffle(base_ids)
        limit=config.get('per_kind_limit',len(base_ids))
        for e in group:
            context=e.get('context_length','legacy')
            context_limit=config.get('per_context_limit',{}).get(context,limit)
            if e.get('base_id',e['id']) in set(base_ids[:context_limit]): episodes.append(e)
    rng.shuffle(episodes)
    if not episodes: raise ValueError('no episodes')
    if config['backend']=='deepseek' and (config.get('max_calls',0)<1 or not config.get('model') or config.get('max_tokens',0)<1):
        raise ValueError('explicit paid-call/model/output limits required')
    manifest=dict(schema_version='continuous-tools-v1',config=config,data_sha256=digest(path),code_sha256=code_hash(),
        episode_ids=[e['id'] for e in episodes],checkpoints=sum(len(e['checkpoints']) for e in episodes),
        dataset_source=bundle['source'],baseline='native full history without memory tools/archive',
        setup='scripted actual file reads; no setup LLM calls; checkpoints are continuous agent execution',
        protocol='chat',thinking='disabled',json_output=True)
    manifest.update(base_scenarios=len({e.get('base_id',e['id']) for e in episodes}),
        context_counts=dict(Counter(e.get('context_length','legacy') for e in episodes)),
        expected_episode_checkpoints={e['id']:len(e['checkpoints']) for e in episodes})
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
        pairs=[dict(document_id=r.get('base_id',r['episode_id']),delta=int(r['task_success'])-int(baseline[(r['episode_id'],r['checkpoint'])]['task_success'])) for r in rows if (r['episode_id'],r['checkpoint']) in baseline]
        complete_episodes=[]
        for episode_id in manifest['episode_ids']:
            rs=[r for r in rows if r['episode_id']==episode_id]
            if len(rs)==3: complete_episodes.append(all(r['task_success'] for r in rs))
        agree=[r['format_ok'] and baseline[(r['episode_id'],r['checkpoint'])]['format_ok'] and r['answer']==baseline[(r['episode_id'],r['checkpoint'])]['answer'] for r in rows if (r['episode_id'],r['checkpoint']) in baseline]
        run=dict(mode=mode,checkpoints=len(rows),expected_checkpoints=manifest['checkpoints'],
            completed_length_conditions=len(complete_episodes),
            independent_base_scenarios=len({r.get('base_id',r['episode_id']) for r in rows}),
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
            request_history_modes=dict(Counter(r['history_mode'] for r in responses)),
            request_history_reasons=dict(Counter(r['history_reason'] for r in responses)),
            recovery_enabled_requests=sum(r.get('recovery_tools_enabled',False) for r in responses),
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
    episodes=episode_accounting(records,manifest)
    return dict(manifest=manifest,runs=runs,episodes=episodes,
        by_context=context_accounting(episodes,offline),
        compressed_subset=compressed_accounting(tasks,episodes,offline,manifest['config']['seed']),
        api_attempts=sum(r['model_calls'] for r in runs))


def episode_accounting(records,manifest):
    """Include all started requests, even when a limit interrupts a checkpoint."""
    output=[]
    for mode in manifest['config']['modes']:
        for episode_id in manifest['episode_ids']:
            rs=[r for r in records if r.get('mode')==mode and r.get('episode_id')==episode_id]
            if not rs: continue
            tasks=[r for r in rs if r['event']=='task_result']
            responses=[r for r in rs if r['event']=='response']
            starts=[r for r in rs if r['event']=='start']
            usage=[r['usage'] for r in responses if r.get('usage') is not None]
            expected=manifest.get('expected_episode_checkpoints',{}).get(episode_id,3)
            complete=len(tasks)==expected
            compressed=sum(r['history_mode']=='compressed' for r in responses)
            row=dict(episode_id=episode_id,mode=mode,base_id=rs[0].get('base_id',episode_id),
                context_length=rs[0].get('context_length','legacy'),kind=rs[0].get('kind','unknown'),
                checkpoints=len(tasks),expected_checkpoints=expected,complete=complete,
                episode_success=complete and all(r['task_success'] for r in tasks),
                checkpoint_successes=sum(r['task_success'] for r in tasks),
                model_requests=len(starts),compressed_requests=compressed,
                compression_category='none' if not compressed else 'all' if compressed==len(starts) else 'mixed',
                fallback_requests=sum(r['history_mode']=='fallback_full' for r in responses),
                prompt_tokens=sum(u['prompt_tokens'] for u in usage),completion_tokens=sum(u['completion_tokens'] for u in usage),
                unknown_usage_requests=len(starts)-len(usage),
                total_tokens=sum(u['prompt_tokens']+u['completion_tokens'] for u in usage),
                end_to_end_seconds=sum(r['end_to_end_seconds'] for r in tasks),
                initial_baseline_characters=tasks[0].get('initial_baseline_units') if tasks else None,
                initial_sent_characters=tasks[0]['initial_sent_units'] if tasks else None)
            output.append(row)
    return output


def context_accounting(episodes,offline):
    output=[]
    for context in sorted({r['context_length'] for r in episodes}):
        for mode in sorted({r['mode'] for r in episodes}):
            rows=[r for r in episodes if r['context_length']==context and r['mode']==mode]
            output.append(dict(context_length=context,mode=mode,episodes=len(rows),
                base_scenarios=len({r['base_id'] for r in rows}),complete_episodes=sum(r['complete'] for r in rows),
                episode_success_rate=None if offline else mean([r['episode_success'] for r in rows]),
                checkpoint_success_rate=None if offline else sum(r['checkpoint_successes'] for r in rows)/max(1,sum(r['checkpoints'] for r in rows)),
                actual_compressed_episodes=sum(r['compressed_requests']>0 for r in rows),
                compressed_requests=sum(r['compressed_requests'] for r in rows),requests=sum(r['model_requests'] for r in rows),
                total_tokens=None if offline else sum(r['total_tokens'] for r in rows),
                unknown_usage_requests=0 if offline else sum(r['unknown_usage_requests'] for r in rows),
                initial_characters=mean([r['initial_baseline_characters'] for r in rows]),
                episode_latency=latency_stats(rows)))
    for row in output:
        full=next((r for r in output if r['mode']=='full' and r['context_length']==row['context_length']),None)
        row['total_token_reduction']=None if offline or not full or not full['total_tokens'] else 1-row['total_tokens']/full['total_tokens']
    return output


def compressed_accounting(tasks,episodes,offline,seed):
    """Post-selection diagnostic; baseline is restricted to identical task IDs."""
    full={(r['episode_id'],r['checkpoint']):r for r in tasks if r['mode']=='full'}
    output=[]
    for mode in sorted({r['mode'] for r in episodes}-{'full'}):
        selected=[r for r in episodes if r['mode']==mode and r['compressed_requests']>0]
        ids={r['episode_id'] for r in selected}
        rows=[r for r in tasks if r['mode']==mode and r['episode_id'] in ids]
        paired=[r for r in rows if (r['episode_id'],r['checkpoint']) in full]
        compressed_rows=[r for r in rows if r.get('compressed_requests',0)>0]
        base_episodes=[r for r in episodes if r['mode']=='full' and r['episode_id'] in ids]
        pairs=[dict(document_id=r.get('base_id',r['episode_id']),delta=int(r['task_success'])-int(full[(r['episode_id'],r['checkpoint'])]['task_success'])) for r in paired]
        base_tokens=sum(r['total_tokens'] for r in base_episodes)
        output.append(dict(mode=mode,episodes=len(selected),base_scenarios=len({r['base_id'] for r in selected}),
            complete_episodes=sum(r['complete'] for r in selected),
            episode_success_rate=None if offline else mean([r['episode_success'] for r in selected]),
            all_checkpoints_in_compressed_episodes=len(rows),actually_compressed_checkpoints=len(compressed_rows),
            compressed_checkpoint_success_rate=None if offline else mean([r['task_success'] for r in compressed_rows]),
            compressed_episode_checkpoint_success_rate=None if offline else mean([r['task_success'] for r in rows]),
            matching_full_checkpoint_success_rate=None if offline else mean([full[(r['episode_id'],r['checkpoint'])]['task_success'] for r in paired]),
            matching_full_episode_success_rate=None if offline else mean([r['episode_success'] for r in base_episodes]),
            paired_quality_difference=None if offline else cluster_ci(pairs,'delta',seed),
            total_tokens=None if offline else sum(r['total_tokens'] for r in selected),
            matching_full_total_tokens=None if offline else base_tokens,
            total_token_reduction=None if offline or not base_tokens else 1-sum(r['total_tokens'] for r in selected)/base_tokens,
            interpretation='post-selection diagnostic; includes failures and all calls; not a population accuracy claim'))
    return output


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
    plugin=MemoryPlugin(archive,budget=config['budget_characters'],recent_turns=config['recent_turns'],
        selector=config.get('selector','structured'),min_reduction=config.get('min_reduction',.1))
    if config['backend']=='deepseek' and transport is None:
        transport=DeepSeekChat(config['model'],config['max_tokens'],config.get('timeout',120),json_output=True)
    try:
        with (output/'requests.jsonl').open('x',encoding='utf-8') as log:
            metadata={e['id']:dict(base_id=e.get('base_id',e['id']),kind=e['kind'],
                                  context_length=e.get('context_length','legacy')) for e in episodes}
            def append(record):
                record={**metadata[record['episode_id']],**record}
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
                        snapshots=[s for s in sources if s['tool']=='read_config' and s.get('entity','Alpha')=='Alpha']
                        if episode['kind']=='knowledge_update' and number<2:
                            relevant=[snapshots[-1 if number==0 else 0]['position']]
                        elif episode['kind']=='temporal' and number<2:
                            import re
                            cutoffs=re.findall(r'截至 (\d{4}-\d{2}-\d{2})',checkpoint['query'])
                            relevant=[max((s for s in snapshots if s['observed_at']<=cutoff),key=lambda s:s['observed_at'])['position'] for cutoff in cutoffs]
                        elif episode['kind']=='tool_execution' and number==0:
                            relevant=[s['position'] for s in sources if s['tool']=='read_orders']
                        retained=all(original[p] in prepared[0].messages for p in relevant) if relevant else None
                        row=dict(event='task_result',episode_id=episode['id'],kind=episode['kind'],checkpoint=number,mode=mode,
                            status=result['status'],**score,initial_history_mode=prepared[0].mode,
                            initial_history_reason=prepared[0].reason,initial_sent_units=prepared[0].sent_units,
                            initial_baseline_units=prepared[0].baseline_units,
                            compressed_requests=sum(s['history_mode']=='compressed' for s in result['steps']),
                            fallback_requests=sum(s['history_mode']=='fallback_full' for s in result['steps']),
                            prompt_tokens=result['prompt_tokens'],completion_tokens=result['completion_tokens'],
                            model_calls=result['api_calls'],unknown_usage_requests=result['unknown_usage_requests'],
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
    fields=['episode_id','base_id','context_length','kind','checkpoint','mode','status','format_ok','answer_correct','state_correct',
            'required_tools_completed','constraint_violation','task_success','business_tool_calls','initial_history_mode',
            'initial_history_reason','initial_baseline_units','initial_sent_units','compressed_requests','fallback_requests',
            'prompt_tokens','completion_tokens','model_calls','unknown_usage_requests','source_messages_retained','end_to_end_seconds']
    with (destination/'per_checkpoint.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:
            if r['event']=='task_result': writer.writerow({f:r.get(f) for f in fields})
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    episodes=summary.get('episodes',[])
    if episodes:
        with (destination/'per_episode.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(episodes[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(episodes)
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
