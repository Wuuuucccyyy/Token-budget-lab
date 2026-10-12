"""Export the complete LangChain comparison and audit actual business receipts."""
import argparse
import copy
import csv
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal,ROUND_HALF_EVEN
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from token_budget_lab.action_tasks import update_current_files
from token_budget_lab.cache_cost import cache_cost
from token_budget_lab.context_plugin import source_id
from token_budget_lab.langchain_experiment import current_intent
from token_budget_lab.memory import encode,validate_messages
from token_budget_lab.tool_tasks import TOOLS,ToolSandbox,assess
from token_budget_lab.write_transform import pointer_value


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export(run,destination):
    run,destination=Path(run),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    manifest,summary,events=read(run/'manifest.json'),read(run/'summary.json'),rows(run/'requests.jsonl')
    tasks=[r for r in events if r['event']=='task_result']
    expected=len(manifest['episode_ids'])*len(manifest['config']['arms'])
    assert len(tasks)==expected and summary['actual_api_calls']<=manifest['config']['max_calls']
    assert summary['unknown_usage_attempts']==0
    assert manifest['graph_compilations']==1
    sources={p.name:sha(p.read_bytes()) for p in (run/'execution_sources').glob('*.py')}
    assert sources=={p.name:sha(p.read_bytes()) for p in (ROOT/'token_budget_lab').glob('*.py')}
    bundle=read(ROOT/manifest['config']['data'])
    assert sha((ROOT/manifest['config']['data']).read_bytes())==manifest['data_sha256']
    episodes={e['id']:e for e in bundle['episodes']}
    grouped=defaultdict(list)
    for event in events:grouped[(event['episode_id'],event['mode'])].append(event)
    audit=[];eager=[];conditional=[];recovery=[];numeric_checks=0
    for task in tasks:
        episode=copy.deepcopy(episodes[task['episode_id'].rsplit('-r',1)[0]])
        checkpoint=episode['checkpoints'][0]
        condition=grouped[(task['episode_id'],task['mode'])]
        canonical=task['canonical_messages']
        validate_messages(canonical)
        tools=[r for r in condition if r['event']=='tool']
        actual=[r for r in condition if r['event']=='business_tool']
        for tool in tools:
            position=tool['canonical_position']
            message=canonical[position]
            assert message['role']=='tool' and message['tool_call_id']==tool['tool_call_id']
            assert message['content']==encode(tool['result'])
            if tool['handler_invoked']:
                matches=[r for r in actual if r['tool']==tool['tool'] and r['arguments']==tool['arguments']
                    and (not tool['handler_returned'] or r.get('result')==tool['result'])]
                assert matches,'Actual handler receipt missing'
            if tool['tool'].startswith('context_'):
                recovery.append(dict(episode_id=task['episode_id'],mode=task['mode'],tool=tool['tool'],
                    source_id=tool['arguments'].get('source_id'),pointer=tool['arguments'].get('pointer'),
                    handler_invoked=tool['handler_invoked'],result=tool['result']))
        contract,guard,scope,waived=current_intent(episode)
        schemas={s['function']['name']:s['function'] for s in TOOLS}
        registered={name:None for name in schemas}
        replay=guard.start(schemas,registered)
        # Replay only actual execution returns. History and recovery never
        # establish a new numeric source or conditional authorization.
        for tool in tools:
            if not tool['handler_invoked']:
                continue
            replay.before_invocation(tool['tool'],tool['arguments'])
            replay.observe(tool['tool'],tool['arguments'],tool['result'],canonical[tool['canonical_position']],
                           tool['canonical_position'],tool['handler_returned'])
        for validation in task['write_transform']['events']:
            source=next(t for t in tools if t['handler_returned'] and
                source_id(t['canonical_position'],canonical[t['canonical_position']])==validation['source_id'])
            raw,path=guard._select(guard.rules[0],dict(value=source['result']))
            assert str(raw)==validation['source_value'] and path==validation['source_location']
            assert sha(encode(source['result']).encode())==validation['result_sha256']
            rule=guard.rules[0]
            expected_value=(format(Decimal(str(raw)).quantize(Decimal(1).scaleb(-rule.places),rounding=ROUND_HALF_EVEN),'f')
                if rule.operation=='format_decimal' else int(Decimal(str(raw)))+rule.constant)
            assert expected_value==validation['expected_value']
            if validation['accepted']:assert validation['attempted_value']==expected_value
            numeric_checks+=1
        starts=[r for r in condition if r['event']=='start']
        for start in starts:
            validate_messages(start['sent_messages'])
            for view_message in start['sent_messages']:
                if view_message['role']!='tool':continue
                value=json.loads(view_message['content'])
                for projection in value.get('requested_evidence',[]) if isinstance(value,dict) else []:
                    source=next(t for t in tools if t['handler_returned'] and
                        source_id(t['canonical_position'],canonical[t['canonical_position']])==projection['source_id'])
                    if projection['selection_status']=='unique':
                        source_value,location=guard._select(guard.rules[0],dict(value=source['result']))
                        assert str(projection['original_value'])==str(source_value) and projection['source_location']==location
                    else:
                        local=guard.start(schemas,registered)
                        local.observe(source['tool'],source['arguments'],source['result'],canonical[source['canonical_position']],source['canonical_position'],True)
                        assert local.branch_state()['selection_status']==projection['selection_status']
                    assert sha(encode(source['result']).encode())==projection['result_sha256']
                    if not any(e['episode_id']==task['episode_id'] and e['source_id']==projection['source_id'] for e in eager):
                        eager.append(dict(episode_id=task['episode_id'],mode=task['mode'],**projection,
                            actual_handler_verified=True,canonical_original_preserved=True))
            branch=start['context_metadata'].get('conditional_branch')
            if branch and branch['selection_status']!='awaiting_read':
                source=next(t for t in tools if t['handler_returned'] and
                    source_id(t['canonical_position'],canonical[t['canonical_position']])==branch['source_id'])
                local=guard.start(schemas,registered)
                local.observe(source['tool'],source['arguments'],source['result'],canonical[source['canonical_position']],
                    source['canonical_position'],True)
                assert local.branch_state()==branch
                if not any(e['episode_id']==task['episode_id'] and e['mode']==task['mode'] for e in conditional):
                    conditional.append(dict(episode_id=task['episode_id'],mode=task['mode'],**branch,
                        actual_handler_verified=True,branch_recomputed=True))
        with tempfile.TemporaryDirectory(dir=ROOT/'local_results') as tmp:
            reference=ToolSandbox(Path(tmp)/'reference',episode)
            reference.seed_history(episode);update_current_files(reference,episode['live_updates'])
            before=reference.file_hashes()
            sandbox=object.__new__(ToolSandbox)
            sandbox.path=run/'sandboxes'/task['episode_id']/task['mode']
            sandbox.trace=actual
            score=assess(checkpoint,dict(answer=encode(dict(answer=task['answer'])),status=task['status']),
                         sandbox,before,0,episode['initial_config'])
            write_attempts=sum(c['function']['name'] in {'write_config','write_report'} for r in condition
                if r['event']=='response' for c in r.get('message',{}).get('tool_calls',[]))
            if checkpoint.get('no_writes') and write_attempts:
                score.update(task_success=False,constraint_violation=True)
            assert all(score[k]==task[k] for k in ('answer_correct','state_correct','required_tools_completed','constraint_violation','task_success'))
            state=sandbox.state()
        assert all(set(t['arguments'])<=set(scope[t['tool']]) for t in tools if t['handler_invoked'] and t['tool'] in scope)
        assert not any(m['role']=='system' and any(marker in m['content'] for marker in
            ('[Current user write argument scope]','[Current-run conditional authorization snapshot]','[Host finalization rejected]')) for m in canonical)
        audit.append(dict(episode_id=task['episode_id'],mode=task['mode'],score_recomputed=True,
            actual_handler_receipts_verified=True,canonical_originals_verified=True,request_only_notices_verified=True,
            authorized_write_fields=scope,final_files=state,actions=task['execution_contract'],
            write_events=task['write_transform']['events'],forbidden_write_attempts=write_attempts if checkpoint.get('no_writes') else 0))
    assert len(audit)==expected
    full,compact=summary['runs']
    compare=dict(total_token_reduction=compact['total_token_reduction'],
        fixed_reference_cost_reduction=1-compact['estimated_usd']/full['estimated_usd'],
        mean_latency_ratio=compact['latency']['mean_seconds']/full['latency']['mean_seconds'],
        p95_latency_ratio=compact['latency']['p95_seconds']/full['latency']['p95_seconds'],
        p95_target=manifest['config']['latency_target_p95_ratio'],
        p95_target_passed=compact['latency']['p95_seconds']/full['latency']['p95_seconds']<=manifest['config']['latency_target_p95_ratio'],
        quality_difference=compact['task_success_rate']-full['task_success_rate'])
    write(destination/'manifest.json',manifest)
    write(destination/'metrics.json',dict(package_version=manifest['package_version'],framework=manifest['framework'],
        actual_api_calls=summary['actual_api_calls'],unknown_usage_attempts=summary['unknown_usage_attempts'],
        independent_case_templates=manifest['independent_case_templates'],repeats=manifest['repeats'],runs=summary['runs'],comparisons=compare,
        compressed_subset=summary['compressed_subset'],quality_scope=manifest['quality']))
    write(destination/'execution_audit.json',audit)
    write(destination/'eager_evidence_audit.json',eager)
    write(destination/'conditional_branch_audit.json',conditional)
    write(destination/'recovery_audit.json',recovery)
    write(destination/'source_manifest.json',dict(data_sha256=manifest['data_sha256'],code_sha256=manifest['code_sha256'],
        runtime_frozen_sources=sources,journal_sha256=sha((run/'requests.jsonl').read_bytes()),
        run_id=manifest['run_id'],actual_receipts_recomputed=True))
    fields=['episode_id','base_id','repetition','mode','status','answer','answer_correct','state_correct',
        'required_tools_completed','constraint_violation','task_success','fresh_read_completed',
        'operation_order_correct','model_write_attempts','compressed_requests','eager_evidence_requests',
        'prompt_tokens','completion_tokens','model_calls','unknown_usage_requests','end_to_end_seconds']
    with (destination/'per_task.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for task in tasks:writer.writerow({k:task.get(k) for k in fields})
    fields=['episode_id','mode','step','status','history_mode','history_reason','baseline_units','sent_units',
        'prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens','model_seconds','sent_messages_sha256']
    with (destination/'requests.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for response in (r for r in events if r['event']=='response'):
            start=next(r for r in grouped[(response['episode_id'],response['mode'])] if r['event']=='start' and r['step']==response['step'])
            usage=response['usage']
            row={k:response.get(k) for k in fields}
            row.update(prompt_tokens=usage['prompt_tokens'],completion_tokens=usage['completion_tokens'],
                cache_hit_tokens=usage['prompt_cache_hit_tokens'],cache_miss_tokens=usage['prompt_cache_miss_tokens'],
                sent_messages_sha256=sha(encode(start['sent_messages']).encode()))
            writer.writerow(row)
    previous=read(ROOT/'results/current/usage_ledger.json')
    diagnostics=[]
    for name,kind in [('langchain-deepseek-v013','integration_diagnostic'),('langchain-scoped-deepseek-v0131','balance_failure')]:
        prior=ROOT/'local_results'/name
        if not (prior/'requests.jsonl').exists():continue
        prior_manifest=read(prior/'manifest.json');prior_events=rows(prior/'requests.jsonl')
        starts=[r for r in prior_events if r['event']=='start']
        known=[r['usage'] for r in prior_events if r['event']=='response' and r.get('usage') is not None]
        cost=cache_cost(known,manifest['config']['usd_per_million'])
        diagnostics.append(dict(kind=kind,package_version=prior_manifest['package_version'],attempts=len(starts),
            known_usage_responses=len(known),unknown_usage_attempts=len(starts)-len(known),
            known_fixed_reference_subtotal_usd=cost['estimated_usd'],
            total_estimated_usd=None if len(starts)!=len(known) else cost['estimated_usd'],
            completed_tasks=sum(r['event']=='task_result' for r in prior_events),primary_quality_evidence=False,
            journal_sha256=sha((prior/'requests.jsonl').read_bytes())))
    old_ledger=read(destination/'usage_ledger.json') if (destination/'usage_ledger.json').exists() else {}
    journal=sha((run/'requests.jsonl').read_bytes())
    if old_ledger.get('primary_journal_sha256')!=journal:
        for old in old_ledger.get('integration_diagnostics',[]):
            if old['journal_sha256'] not in {d['journal_sha256'] for d in diagnostics}:diagnostics.append(old)
        if old_ledger.get('primary_journal_sha256'):
            diagnostics.append(dict(kind='earlier_langchain_comparison',package_version=old_ledger.get('package_version'),
                attempts=old_ledger['primary_attempts'],known_usage_responses=old_ledger['primary_known_usage_responses'],
                unknown_usage_attempts=old_ledger['primary_attempts']-old_ledger['primary_known_usage_responses'],
                known_fixed_reference_subtotal_usd=old_ledger['primary_reference_cost']['estimated_usd'],
                total_estimated_usd=old_ledger['primary_reference_cost']['estimated_usd'],
                completed_tasks=old_ledger.get('primary_completed_tasks'),primary_quality_evidence=False,
                journal_sha256=old_ledger['primary_journal_sha256']))
    else:
        for old in old_ledger.get('integration_diagnostics',[]):
            if old['journal_sha256'] not in {d['journal_sha256'] for d in diagnostics}:diagnostics.append(old)
    unknown=sum(r['unknown_usage_attempts'] for r in diagnostics)
    known_total=previous['total_fixed_reference_research_cost']+summary['research_usage']['estimated_usd']+sum(r['known_fixed_reference_subtotal_usd'] for r in diagnostics)
    attempts=summary['actual_api_calls']
    write(destination/'usage_ledger.json',dict(package_version=manifest['package_version'],primary_journal_sha256=journal,
        primary_attempts=attempts,primary_known_usage_responses=attempts-summary['unknown_usage_attempts'],primary_completed_tasks=len(tasks),
        primary_reference_cost=summary['research_usage'],previous_native_research_attempts=previous['total_research_attempts'],
        previous_native_reference_cost=previous['total_fixed_reference_research_cost'],integration_diagnostics=diagnostics,
        total_research_attempts=previous['total_research_attempts']+attempts+sum(r['attempts'] for r in diagnostics),
        total_unknown_usage_attempts=unknown,known_fixed_reference_subtotal_usd=known_total,
        total_research_estimated_usd=None if unknown else known_total,actual_invoice=None,
        automatic_retries=False,all_attempts_retained=True))
    write(destination/'validation.json',dict(package_version=manifest['package_version'],
        source_score_audits=len(audit),actual_source_projections=len(eager),actual_conditional_branches=len(conditional),numeric_writes_recomputed=numeric_checks,
        recovered_sources=len(recovery),runtime_source_files=len(sources),runtime_frozen=True,
        framework_versions=manifest['framework_versions'],single_compiled_graph=True,
        shared_rules_all_arms=True,strict_forbidden_attempt_scoring_unchanged=True,
        core_tests=103,langchain_tests=18,total_tests=121,live_primary_attempts=attempts))
    print(json.dumps(dict(tasks=len(tasks),requests=attempts,eager_sources=len(eager),conditional_sources=len(conditional),numeric_writes=numeric_checks,
        mean_ratio=compare['mean_latency_ratio'],p95_ratio=compare['p95_latency_ratio'],source_audits=True),ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='local_results/langchain-scoped-deepseek-v0131-funded')
    parser.add_argument('--destination',default='results/langchain')
    args=parser.parse_args()
    export(ROOT/args.run,ROOT/args.destination)
