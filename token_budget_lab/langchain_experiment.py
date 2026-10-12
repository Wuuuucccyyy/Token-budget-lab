"""Toggle context compression on one LangChain graph with complete task accounting."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents import create_agent

from .action_tasks import execution_contract, update_current_files
from .action_contract import ActionContract, RequiredAction
from .cache_cost import cache_cost
from .experiment import ROOT, code_hash
from .langchain_adapter import TokenBudgetContext, TokenBudgetMiddleware, business_tools
from .langchain_model import DeepSeekChatModel, canonical_messages, langchain_messages
from .memory import encode
from .memory_agent import DeepSeekChat
from .tool_experiment import summarize
from .tool_tasks import TOOLS, ToolSandbox, assess
from .transform_tasks import guard_from_query
from . import __version__


def plan(config):
    if config['arms'] != ['full','context_fast'] or config['max_steps'] != 10:
        raise ValueError('Shared Full/context_fast comparison with ten model calls required')
    if config.get('declare_write_arguments') is not True or config.get('conditional_completion') is not True:
        raise ValueError('Explicit shared argument scope and conditional completion required')
    repeats=config['repeats']
    if type(repeats) is not int or not 1<=repeats<=5:
        raise ValueError('One to five repetitions required')
    data=ROOT/config['data']
    bundle=json.loads(data.read_text(encoding='utf-8'))
    episodes=[]
    for repetition in range(1,repeats+1):
        for original in bundle['episodes']:
            current_intent(original)
            episode=copy.deepcopy(original)
            episode.update(id=original['id']+f'-r{repetition:02d}',repetition=repetition)
            episodes.append(episode)
    if config['max_calls'] < len(episodes)*len(config['arms'])*config['max_steps']:
        raise ValueError('Paid cap must cover the declared bounded comparison')
    versions={n:importlib.metadata.version(n) for n in ('langchain','langchain-core','langgraph')}
    manifest=dict(package_version=__version__,framework='langchain.create_agent',framework_versions=versions,
        config={**config,'modes':config['arms'],'backend':'deepseek'},
        episode_ids=[e['id'] for e in episodes],checkpoints=len(episodes),
        independent_case_templates=len(bundle['episodes']),repeats=repeats,
        expected_episode_checkpoints={e['id']:1 for e in episodes},
        data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),code_sha256=code_hash(),
        graph_compilations=1,plugin_switch='TokenBudgetContext.enabled on the same compiled graph',
        shared_controls='same model, business tools, fresh-action contract, numeric/conditional guard, and ten-call cap',
        write_argument_scope='host-declared tool fields from current user intent; no evaluation values or labels',
        conditional_completion='write/check requirements apply only to the actual unique-source branch; fresh read always required',
        graph_max_concurrency=1,graph_recursion_limit=70,max_recovery_rounds=3,
        latency='complete agent.invoke wall time, including host setup, tools, recovery, checks and cleanup; graph compilation excluded',
        cache_control='independent random namespace per task/arm; natural within-task cache; provider cache cannot be cleared',
        dataset_source=bundle['source'],quality='reused synthetic templates; repetitions are not new independent tasks',
        pricing_source='https://api-docs.deepseek.com/quick_start/pricing/',pricing_reference_date='2026-10-12',
        pricing_interpretation='same fixed reference rates as the native-agent comparison; not an account invoice',
        protocol='chat',thinking='disabled',json_output=True)
    return episodes,manifest


def current_intent(episode):
    """Frozen fixture wording adapter; never inspect expected answers or file-state labels."""
    query=episode['checkpoints'][0]['query']
    guard=guard_from_query(query)
    contract=execution_contract(episode['request_actions'])
    conditional=()
    if guard.conditional_read_only:
        rule=guard.rules[0]
        if rule.write_tool!='write_report' or '写入 report.json 的 total' not in query or 'run_checks' not in query:
            raise ValueError('Explicit conditional write/check intent required')
        parent=contract.actions[-1].id
        contract=ActionContract([*contract.actions,
            RequiredAction('conditional_write',rule.write_tool,after=(parent,),result_matches=dict(written=True)),
            RequiredAction('conditional_check','run_checks',after=('conditional_write',),result_matches=dict(passed=True))],
            {**contract.before_tools,rule.write_tool:[parent],'run_checks':['conditional_write']})
        conditional=('conditional_write','conditional_check')
    arguments={rule.write_tool:[rule.value_argument,*rule.fixed_arguments] for rule in guard.rules}
    return contract,guard,arguments,conditional


def execute(config,output,transport=None):
    episodes,manifest=plan(config)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists() or (output/'requests.jsonl').exists():
        raise ValueError('Fresh output required; no automatic replay')
    manifest.update(run_id=uuid.uuid4().hex,started_at_utc=datetime.now(timezone.utc).isoformat())
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    snapshot=output/'execution_sources';snapshot.mkdir()
    for path in (ROOT/'token_budget_lab').glob('*.py'):
        (snapshot/path.name).write_bytes(path.read_bytes())
    client=transport or DeepSeekChat(config['model'],config['max_tokens'],config.get('timeout',120),json_output=True)
    model=DeepSeekChatModel(transport=client,model_name=config['model'])
    tools=business_tools(TOOLS)
    middleware=TokenBudgetMiddleware(tools,output/'archives',budget=config['budget_characters'],
        offload_tools=config['offload_tools'],offload_threshold=config['offload_threshold'],
        max_steps=config['max_steps'],max_recovery_rounds=3,eager_evidence=True)
    # Compiled once. Each invocation supplies independent handlers and policy;
    # enabled is the only difference in the execution framework between arms.
    agent=create_agent(model,tools=tools,middleware=[middleware],context_schema=TokenBudgetContext)
    records=[];attempts=0
    rng=random.Random(config['seed']);rng.shuffle(episodes)
    try:
        with (output/'requests.jsonl').open('x',encoding='utf-8') as log:
            def append(row):
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
                records.append(row)
            for episode in episodes:
                arms=config['arms'].copy();rng.shuffle(arms)
                for arm in arms:
                    active=[False]
                    meta=dict(episode_id=episode['id'],base_id=episode['base_id'],kind=episode['kind'],
                        repetition=episode['repetition'],context_length='long',mode=arm,checkpoint=0)
                    def action(row):
                        if active[0]:append(dict(event='business_tool',**meta,**row))
                    sandbox=ToolSandbox(output/'sandboxes'/episode['id']/arm,episode,action)
                    history,_=sandbox.seed_history(episode)
                    update_current_files(sandbox,episode['live_updates'])
                    history[0]['content']='[Benchmark namespace; no task facts] '+hashlib.sha512(
                        encode([manifest['run_id'],episode['id'],arm]).encode()).hexdigest()+'\n'+history[0]['content']
                    checkpoint=episode['checkpoints'][0]
                    history.append(dict(role='user',content=checkpoint['query']))
                    initial=[];steps=[]
                    def observe(row):
                        nonlocal attempts
                        if row['event']=='start':
                            if attempts>=config['max_calls']:raise RuntimeError('Paid-call limit reached')
                            attempts+=1
                            if not initial:initial.append(row)
                        elif row['event']=='response':steps.append(row)
                        append(dict(**row,**meta))
                    policy,guard,scope,conditional=current_intent(episode)
                    context=TokenBudgetContext('langchain',episode['id']+':'+arm,enabled=arm=='context_fast',
                        handlers=sandbox.handlers(),action_contract=policy,write_guard=guard,
                        write_arguments=scope,conditional_obligations=conditional,observer=observe)
                    trace_start,hashes=len(sandbox.trace),sandbox.file_hashes()
                    active[0]=True
                    started=time.perf_counter()
                    state=agent.invoke(dict(messages=langchain_messages(history)),context=context,
                        config=dict(max_concurrency=1,recursion_limit=70))
                    elapsed=time.perf_counter()-started
                    result=dict(**state['token_budget_result'],messages=canonical_messages(state['messages']),steps=steps)
                    score=assess(checkpoint,result,sandbox,hashes,trace_start,episode['initial_config'])
                    write_attempts=sum(c['function']['name'] in {'write_config','write_report'}
                        for m in result['messages'][len(history):] for c in m.get('tool_calls',[]))
                    if checkpoint.get('no_writes') and write_attempts:
                        score.update(constraint_violation=True,task_success=False)
                    required=episode['request_actions']['required'][0]
                    actual=sandbox.trace[trace_start:]
                    fresh=any(a['success'] and a['tool']==required['tool'] and a['arguments']==required['arguments'] for a in actual)
                    writes=[i for i,a in enumerate(actual) if a['success'] and a['tool'] in {'write_config','write_report'}]
                    checks=[i for i,a in enumerate(actual) if a['success'] and a['tool']=='run_checks' and a['result'].get('passed')]
                    ordered=not ('run_checks' in checkpoint['required'] and writes) or bool(checks and checks[-1]>writes[-1])
                    score['required_tools_completed']=score['required_tools_completed'] and fresh
                    score['task_success']=score['task_success'] and fresh and ordered
                    append(dict(event='task_result',**meta,**score,status=result['status'],fresh_read_completed=fresh,
                        operation_order_correct=ordered,model_write_attempts=write_attempts,
                        execution_contract=result['action_contract'],write_transform=result['write_transform'],
                        initial_history_mode=initial[0]['history_mode'],initial_history_reason=initial[0]['history_reason'],
                        initial_sent_units=initial[0]['sent_units'],initial_baseline_units=initial[0]['baseline_units'],
                        source_messages_retained=None,compressed_requests=result['compressed_requests'],fallback_requests=0,
                        eager_evidence_requests=sum(bool(s['context_metadata'].get('eager_evidence_source_ids')) for s in steps),
                        prompt_tokens=result['prompt_tokens'],completion_tokens=result['completion_tokens'],
                        model_calls=result['api_calls'],unknown_usage_requests=result['unknown_usage_requests'],
                        end_to_end_seconds=elapsed,canonical_messages=result['messages']))
                    print(json.dumps(dict(episode=episode['id'],mode=arm,passed=score['task_success'],
                        model_calls=result['api_calls'],total_attempts=attempts)),flush=True)
    finally:
        middleware.close()
        summary=summarize(records,manifest)
        for run in summary['runs']:
            usage=[r['usage'] for r in records if r['event']=='response' and r['mode']==run['mode'] and r.get('usage') is not None]
            run.update(cache_cost(usage,config['usd_per_million']))
            if run['unknown_usage_requests']:
                run['known_usage_subtotal_usd']=run['estimated_usd']
                run['estimated_usd']=None
            tasks=[r for r in records if r['event']=='task_result' and r['mode']==run['mode']]
            run['fresh_read_completed']=sum(r['fresh_read_completed'] for r in tasks)
            run['write_transform_rejected']=sum(r['write_transform']['rejected'] for r in tasks)
        summary['actual_api_calls']=attempts
        summary['unknown_usage_attempts']=attempts-sum(r['event']=='response' and r.get('usage') is not None for r in records)
        summary['research_usage']=cache_cost([r['usage'] for r in records if r['event']=='response' and r.get('usage') is not None],config['usd_per_million'])
        if summary['unknown_usage_attempts']:summary['research_usage']['estimated_usd']=None
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/langchain_deepseek.json')
    parser.add_argument('--output',default='local_results/langchain-scoped-deepseek-v0131')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if args.plan:
        print(json.dumps(plan(config)[1],ensure_ascii=True,indent=2))
    else:
        print(json.dumps(execute(config,ROOT/args.output),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
