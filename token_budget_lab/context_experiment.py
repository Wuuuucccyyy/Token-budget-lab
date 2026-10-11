"""Four-arm context pilot; isolated initial prefixes and explicit warm probes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .cache_cost import cache_cost
from .context_plugin import ContextPlugin
from .experiment import ROOT, code_hash, stable_hash
from .memory import HistoryArchive, encode
from .memory_agent import DeepSeekChat, run_agent
from .provenance import HistoricalWriteGuard
from .tool_experiment import summarize, export
from .tool_tasks import TOOLS, ToolSandbox, assess


ARMS = ['full','full_guarded','context','context_guarded']


def plan(config):
    if config['arms'] != ARMS or config['max_steps'] < 1 or config['max_calls'] < 1:
        raise ValueError('explicit four arms, steps and total paid-call cap required')
    if not config.get('model') or config.get('max_tokens',0) < 1:
        raise ValueError('explicit model and output limit required')
    path = ROOT/config['data']
    bundle = json.loads(path.read_text(encoding='utf-8'))
    return bundle['episodes'],dict(config={**config,'modes':config['arms'],'backend':'deepseek'},
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_sha256=code_hash(),
        dataset_source=bundle['source'],episode_ids=[e['id'] for e in bundle['episodes']],
        checkpoints=sum(len(e['checkpoints']) for e in bundle['episodes']),
        expected_episode_checkpoints={e['id']:len(e['checkpoints']) for e in bundle['episodes']},
        quality='all model-driven trajectories over synthetic fixtures; no gold labels in inference; synthetic pilot only',
        cache_control='unique 128-hex system prefix per episode/arm; natural within-task caching; cannot flush provider cache',
        warm_probe='exact first request replayed once after each episode; no tool execution/quality scoring; paid usage separate',
        pricing_source='https://api-docs.deepseek.com/quick_start/pricing/',pricing_verified_date='2026-10-12',
        pricing_interpretation='fixed off-peak USD rates for comparable repricing, not actual invoice; peak rates twice these',
        baseline='full native history; full_guarded adds same provenance validation as context_guarded',
        protocol='chat',thinking='disabled',json_output=True)


def execute(config, output, transport=None):
    episodes,manifest=plan(config)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() or (output/'requests.jsonl').exists():
        raise ValueError('fresh output required; no automatic paid replay')
    manifest['run_id']=uuid.uuid4().hex
    manifest['started_at_utc']=datetime.now(timezone.utc).isoformat()
    manifest['identity']=stable_hash(manifest)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'execution_sources').mkdir(exist_ok=True)
    for p in (ROOT/'token_budget_lab').glob('*.py'):
        (output/'execution_sources'/p.name).write_bytes(p.read_bytes())
    records=[];attempts=0
    rng=random.Random(config['seed'])
    rng.shuffle(episodes)
    client=transport or DeepSeekChat(config['model'],config['max_tokens'],config.get('timeout',120),json_output=True)
    archive=HistoryArchive(output/'history.sqlite3')
    plugins={arm:ContextPlugin(archive,budget=config['budget_characters'],recent_turns=1,
        offload_tools=config['offload_tools'],offload_threshold=config['offload_threshold']) for arm in ARMS if arm.startswith('context')}
    summary=None
    try:
        with (output/'requests.jsonl').open('x',encoding='utf-8') as log:
            def append(row):
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno());records.append(row)
            for episode in episodes:
                arms=ARMS.copy();rng.shuffle(arms)
                for arm in arms:
                    phase=[-1]
                    def action(row):
                        if phase[0] >= 0:append(dict(event='business_tool',episode_id=episode['id'],mode=arm,checkpoint=phase[0],**row))
                    sandbox=ToolSandbox(output/'sandboxes'/episode['id']/arm,episode,action)
                    history,_=sandbox.seed_history(episode)
                    nonce=hashlib.sha512(encode([manifest['run_id'],episode['id'],arm]).encode()).hexdigest()
                    history[0]['content']='[Benchmark namespace; no task facts] '+nonce+'\n'+history[0]['content']
                    first_request=[]
                    previous=[]
                    for number,checkpoint in enumerate(episode['checkpoints']):
                        phase[0]=number
                        history.append(dict(role='user',content=checkpoint['query']))
                        guard=HistoricalWriteGuard(history) if arm.endswith('_guarded') and 'timeout_ms' in checkpoint['query'] else None
                        trace_start=len(sandbox.trace);before_hashes=sandbox.file_hashes();prepared=[]
                        def before(step,p):
                            nonlocal attempts,previous
                            if attempts >= config['max_calls']:raise RuntimeError('paid call limit reached')
                            common=0
                            for a,b in zip(previous,p.messages):
                                if a != b:break
                                common+=1
                            p.context_metadata['actual_common_prefix_messages']=common
                            p.context_metadata['previous_actual_messages']=len(previous)
                            previous=copy.deepcopy(p.messages)
                            append(dict(event='start',episode_id=episode['id'],base_id=episode['base_id'],context_length='long',mode=arm,
                                checkpoint=number,step=step,sent_messages_sha256=stable_hash(p.messages),started_at_utc=datetime.now(timezone.utc).isoformat()))
                            attempts+=1
                            if step == 0:prepared.append(p)
                        class Capture:
                            last={}
                            def __call__(self,messages,tools):
                                if not first_request:first_request.append((copy.deepcopy(messages),copy.deepcopy(tools)))
                                try:return client(messages,tools)
                                finally:self.last=client.last
                        def response(row):append(dict(event='response',episode_id=episode['id'],base_id=episode['base_id'],context_length='long',mode=arm,checkpoint=number,**row))
                        def tools(row):append(dict(event='recovery_guard',episode_id=episode['id'],mode=arm,checkpoint=number,**row))
                        result=run_agent(history,Capture(),plugins.get(arm),'context',episode['id']+':'+arm,
                            mode='recover' if arm.startswith('context') else 'full',memory_tools=arm.startswith('context'),
                            max_steps=config['max_steps'],max_recovery_rounds=3,extra_tools=TOOLS,handlers=sandbox.handlers(),
                            snapshot_read_tools=config['snapshot_read_tools'],write_guard=guard,
                            before_request=before,after_response=response,after_tools=tools)
                        history=result['messages']
                        score=assess(checkpoint,result,sandbox,before_hashes,trace_start,episode['initial_config'])
                        append(dict(event='task_result',episode_id=episode['id'],base_id=episode['base_id'],context_length='long',kind=episode['kind'],mode=arm,
                            checkpoint=number,status=result['status'],**score,
                            initial_history_mode=prepared[0].mode,initial_history_reason=prepared[0].reason,
                            initial_sent_units=prepared[0].sent_units,initial_baseline_units=prepared[0].baseline_units,
                            compressed_requests=sum(s['history_mode']=='compressed' for s in result['steps']),
                            fallback_requests=sum(s['history_mode']=='fallback_full' for s in result['steps']),
                            source_messages_retained=None,prompt_tokens=result['prompt_tokens'],completion_tokens=result['completion_tokens'],
                            model_calls=result['api_calls'],unknown_usage_requests=result['unknown_usage_requests'],end_to_end_seconds=result['end_to_end_seconds']))
                        if result['status']=='error':raise RuntimeError('API failure retained; no retry')
                    if config.get('warm_probe',True):
                        if attempts >= config['max_calls']:raise RuntimeError('paid call limit reached before probe')
                        attempts+=1
                        messages,tools=first_request[0]
                        append(dict(event='cache_probe_start',episode_id=episode['id'],mode=arm,checkpoint=-1,
                                    sent_messages_sha256=stable_hash(messages),tools_sha256=stable_hash(tools)))
                        started=time.perf_counter()
                        try:
                            replay=client(messages,tools)
                            append(dict(event='cache_probe',episode_id=episode['id'],mode=arm,checkpoint=-1,
                                usage=replay.get('usage'),finish_reason=replay.get('finish_reason'),end_to_end_seconds=time.perf_counter()-started,
                                tools_executed=False))
                        except Exception:
                            append(dict(event='cache_probe',episode_id=episode['id'],mode=arm,checkpoint=-1,
                                usage=client.last.get('usage'),status='error',tools_executed=False))
                            raise RuntimeError('probe failed; no automatic replay') from None
    finally:
        archive.close()
        summary=summarize(records,manifest)
        rates=config['usd_per_million']
        for run in summary['runs']:
            usages=[r['usage'] for r in records if r['event']=='response' and r['mode']==run['mode'] and r.get('usage') is not None]
            run.update(cache_cost(usages,rates))
            if run['unknown_usage_requests']:run['estimated_usd']=None
        probe=[r for r in records if r['event']=='cache_probe']
        summary['cache_probes']={a:cache_cost([r.get('usage') or {} for r in probe if r['mode']==a],rates) for a in ARMS}
        summary['total_api_attempts_including_probes']=attempts
        summary['probe_api_attempts']=sum(r['event']=='cache_probe_start' for r in records)
        summary['research_usage_including_probes']=cache_cost([r.get('usage') or {} for r in records if r['event'] in {'response','cache_probe'}],rates)
        known=sum(r.get('usage') is not None for r in records if r['event'] in {'response','cache_probe'})
        summary['unknown_usage_attempts_including_probes']=attempts-known
        if attempts != known:summary['research_usage_including_probes']['estimated_usd']=None
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--plan',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.plan:
        _,manifest=plan(config);print(json.dumps(manifest,ensure_ascii=True,indent=2));return
    if not args.output:p.error('--output required for paid execution')
    execute(config,args.output)


if __name__=='__main__':main()
