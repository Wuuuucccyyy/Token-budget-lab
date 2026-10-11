"""Accounting and exports for current plugin task trajectories."""
import csv
import json
from collections import Counter
from pathlib import Path
from .analysis import cluster_ci, latency_stats, mean

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
            if len(rs)==manifest.get('expected_episode_checkpoints',{}).get(episode_id,3):
                complete_episodes.append(all(r['task_success'] for r in rs))
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
            recovery_tool_calls=sum(c['function']['name'].startswith(('history_','context_')) for r in responses for c in r.get('message',{}).get('tool_calls',[])),
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

