"""Export all tasks, paid requests and host checks of the current pilot."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from token_budget_lab.memory import encode
from token_budget_lab.tool_experiment import export
from token_budget_lab.write_transform import FreshWriteGuard, WriteTransform, WriteValueRejected
from token_budget_lab.context_plugin import source_id


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def export_current(run,output,diagnostic=None):
    run,output=Path(run),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    manifest=summary['manifest']
    journal=(run/'requests.jsonl').read_bytes()
    rows=[json.loads(x) for x in journal.decode('utf-8').splitlines()]
    journal_sha=hashlib.sha256(journal).hexdigest()
    review_path=run/'review.json'
    review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else None
    reviewed=bool(review and review['journal_sha256']==journal_sha)
    tasks=[r for r in rows if r['event']=='task_result']
    responses=[r for r in rows if r['event']=='response']
    # Generic export goes to a local directory; only focused current artifacts
    # below are placed in the public repository.
    export(run,run/'public_export')
    for name in ('per_checkpoint.csv','final_answers.csv'):
        (output/name).write_bytes((run/'public_export'/name).read_bytes())
    fields=['episode_id','mode','step','status','history_mode','prompt_tokens','completion_tokens',
            'prompt_cache_hit_tokens','prompt_cache_miss_tokens','tools','end_to_end_seconds','context_metadata']
    with (output/'per_request.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in responses:
            usage=row.get('usage') or {}
            writer.writerow({**{k:row.get(k) for k in fields[:5]},
                **{k:usage.get(k) for k in fields[5:9]},
                'tools':json.dumps([c['function']['name'] for c in row.get('message',{}).get('tool_calls',[])]),
                'end_to_end_seconds':row['end_to_end_seconds'],
                'context_metadata':json.dumps(row['context_metadata'],sort_keys=True)})
    audits=[]
    for task in tasks:
        actions=[r for r in rows if r['event']=='business_tool' and r['episode_id']==task['episode_id'] and r['mode']==task['mode']]
        receipts=task['execution_contract']['receipts']
        matched=len(actions)==len(receipts) and all(
            a['tool']==r['tool'] and a['arguments']==r['arguments'] and a['success']==r['handler_returned']
            and (not a['success'] or hashlib.sha256(encode(a['result']).encode()).hexdigest()==r['result_sha256'])
            for a,r in zip(actions,receipts))
        if not matched:
            raise ValueError('execution receipts differ from actual handler journal')
        audits.append(dict(episode_id=task['episode_id'],mode=task['mode'],status=task['status'],task_success=task['task_success'],
            fresh_read_completed=task['fresh_read_completed'],execution_contract=task['execution_contract'],receipts_match_business_calls=True))
    if 'context_transform' in manifest['config']['arms']:
        audit_transforms(rows,tasks,manifest,output)
    write(output/'execution_audit.json',audits)
    keys=('mode','checkpoints','task_success_rate','answer_accuracy','artifact_state_accuracy','fresh_read_completed',
          'model_calls','recovery_tool_calls','prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens',
          'estimated_usd','latency','status_counts','constraint_violations','write_transform_accepted','write_transform_rejected')
    metrics=dict(measured_at=manifest['started_at_utc'],package_version='0.10.0' if 'context_transform' in manifest['config']['arms'] else '0.9.0',case_count=manifest['checkpoints'],
        arms=manifest['config']['arms'],scope=manifest['dataset_source'],
        max_steps_per_task=manifest['config']['max_steps'],actual_api_calls=summary['total_api_attempts'],
        unknown_usage_requests=summary['unknown_usage_attempts'],source_package_sha256=manifest['code_sha256'],
        data_sha256=manifest['data_sha256'],rates=manifest['config']['usd_per_million'],
        price_interpretation=manifest['pricing_interpretation'],
        runs=[{k:r[k] for k in keys if k in r} for r in summary['runs']],
        reviewed_findings=review['findings'] if reviewed else [])
    write(output/'metrics.json',metrics)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'execution_sources').glob('*.py')}
    write(output/'source_manifest.json',dict(code_sha256=manifest['code_sha256'],files=sources,
        note='Exact package bytes frozen before calls; frozen full copies retained in the private run directory'))
    write(output/'usage_ledger.json',dict(journal_sha256=hashlib.sha256(journal).hexdigest(),
        total_attempts=summary['total_api_attempts'],known_responses=len([r for r in responses if r.get('usage') is not None]),
        unknown_usage_attempts=summary['unknown_usage_attempts'],max_steps=6,paid_call_cap=manifest['config']['max_calls'],
        automatic_retries=False,includes_all_failures=True,research_cost=summary['research_usage'],actual_invoice=None))
    diagnostic_note=None
    if diagnostic:
        diagnostic=Path(diagnostic)
        d=json.loads((diagnostic/'summary.json').read_text(encoding='utf-8'))
        ledger=json.loads((output/'usage_ledger.json').read_text(encoding='utf-8'))
        ledger['protocol_diagnostic']=dict(api_attempts=d['total_api_attempts'],unknown_usage_attempts=d['unknown_usage_attempts'],
            fixed_reference_research_cost=d['research_usage']['estimated_usd'],
            journal_sha256=hashlib.sha256((diagnostic/'requests.jsonl').read_bytes()).hexdigest(),
            reason='conditional obligations exposed expected write branch; scene labels present in request IDs; not primary quality evidence')
        ledger['total_research_attempts_including_diagnostic']=ledger['total_attempts']+d['total_api_attempts']
        ledger['total_fixed_reference_research_cost']=ledger['research_cost']['estimated_usd']+d['research_usage']['estimated_usd']
        write(output/'usage_ledger.json',ledger)
        diagnostic_note=(f"另保留 {d['total_api_attempts']} 次协议诊断：条件题的执行合同提前提示写入分支，场景标签进入了文件 request_id。修正后所有条件题只声明同样的必需读取，条件写入由模型根据原文判断，场景 ID 改为无语义编号，并在评分阶段核对写后检查顺序。诊断质量不作为主成绩。两轮共 {ledger['total_research_attempts_including_diagnostic']} 次调用，固定参考价格总估算约 ${ledger['total_fixed_reference_research_cost']:.6f}；没有删除前轮消耗。")
    groups={}
    for mode in manifest['config']['arms']:
        selected=[t for t in tasks if t['mode']==mode]
        new=lambda t:t.get('evaluation_group')=='new_boundary' or ('evaluation_group' not in t and t['episode_id'].startswith('locator-'))
        groups[mode]=dict(reused_cases=dict(cases=sum(not new(t) for t in selected),
            passed=sum(t['task_success'] for t in selected if not new(t))),
            new_boundary_cases=dict(cases=sum(new(t) for t in selected),
            passed=sum(t['task_success'] for t in selected if new(t))),
            actually_compressed=dict(cases=sum(t['compressed_requests']>0 for t in selected),
                passed=sum(t['task_success'] for t in selected if t['compressed_requests']>0)))
    write(output/'case_groups.json',groups)
    if 'context_transform' in manifest['config']['arms']:
        return transform_report(summary,metrics,groups,output)
    by={r['mode']:r for r in metrics['runs']}
    total=lambda r:r['prompt_tokens']+r['completion_tokens']
    comparisons=[]
    for base in ('context_lexical','full'):
        a,b=by['context_structured'],by[base]
        comparisons.append(dict(variant='context_structured',baseline=base,
            task_quality_delta=a['task_success_rate']-b['task_success_rate'],
            total_token_reduction=1-total(a)/total(b),fixed_reference_cost_reduction=1-a['estimated_usd']/b['estimated_usd'],
            p95_seconds_ratio=a['latency']['p95_seconds']/b['latency']['p95_seconds']))
    write(output/'comparisons.json',comparisons)
    def table(name,fn):
        return '| '+name+' | '+' | '.join(fn(by[m]) for m in manifest['config']['arms'])+' |'
    report=['# 精准证据定位的真实对照','',
        'v0.9，12 个长历史合成情境、三组、同轮随机顺序。六个为已有诊断情境，六个新增数组打乱、稀疏编号、超长对象、重复事件、缺失事件与文本末尾目标。包含重复模板，不是独立自然任务确认。','',
        '| 指标 | Full + 执行检查 | 原恢复方案 + 执行检查 | 精准定位 + 执行检查 |',
        '|---|---:|---:|---:|',
        table('完整任务通过',lambda r:f"{round(r['task_success_rate']*12)}/12"),
        table('指定本轮读取',lambda r:f"{r['fresh_read_completed']}/12"),
        table('答案准确率',lambda r:f"{r['answer_accuracy']:.1%}"),
        table('文件状态准确率',lambda r:f"{r['artifact_state_accuracy']:.1%}"),
        table('模型调用',lambda r:str(r['model_calls'])),
        table('恢复调用',lambda r:str(r['recovery_tool_calls'])),
        table('输入输出总 token',lambda r:f"{total(r):,}"),
        table('固定参考价格 USD',lambda r:f"{r['estimated_usd']:.6f}"),
        table('整任务 P95 秒',lambda r:f"{r['latency']['p95_seconds']:.3f}"),'',
        '## 本轮结论','',
        f"原方案完整任务通过 {round(by['context_lexical']['task_success_rate']*12)}/12，精准定位通过 {round(by['context_structured']['task_success_rate']*12)}/12；恢复调用分别为 {by['context_lexical']['recovery_tool_calls']} 和 {by['context_structured']['recovery_tool_calls']} 次；context_stalled 分别为 {by['context_lexical']['status_counts'].get('context_stalled',0)} 和 {by['context_structured']['status_counts'].get('context_stalled',0)} 次。每任务仍最多六次模型请求、三轮恢复，没有修改中间状态或删去失败。",'',
        f"Full 通过 {round(by['full']['task_success_rate']*12)}/12。精准定位与 Full 的逐任务答案一致率为 {next(r['agreement_with_full'] for r in summary['runs'] if r['mode']=='context_structured'):.1%}。总体通过率与逐题答案一致率分别评估，当前任务质量仍有差距。十二个情境是单次、包含重复模板的诊断，不能推断一般任务非劣性。",'',
        '## 剩余问题与下一步','',
        *([paragraph for finding in review['findings'] for paragraph in (finding,'')]
        if reviewed else ['本次日志尚未人工核对失败轨迹。不能沿用另一轮的失败原因；应结合当前逐任务成绩、实际 handler 和恢复结果检查数值变换、来源与步骤预算。','']),
        '本轮没有实现自动时序来源筛选或本轮写入值核验。context_find 只按调用方指定的源定位；全局 context_search 仍保留不同版本，不能自动判定有效版本。','',
        '## 质量、成本与时间','']
    for c in comparisons:
        report.append(f"- 精准定位相对 {c['baseline']}：总 token 减少 {c['total_token_reduction']:.1%}，固定参考价格费用减少 {c['fixed_reference_cost_reduction']:.1%}，P95 为其 {c['p95_seconds_ratio']:.2f} 倍。")
    report += ['',
        f"三组共 {summary['total_api_attempts']} 次真实调用，未知 usage {summary['unknown_usage_attempts']} 次。固定参考价格合计约 ${summary['research_usage']['estimated_usd']:.6f}，不是账户账单或当前实时价格。全部恢复、失败与最终回答计入；没有额外缓存重放。",'',
        *([diagnostic_note,''] if diagnostic_note else []),
        '两种插件的全部十二个情境均实际压缩，压缩子集成功率与各自全组相同。已有六个情境与六个新增边界情境分别统计，见 [case_groups.json](case_groups.json)。','',
        '## 协议与可核对资料','',
        '三组均使用同一执行合同、实际业务工具和评分条件。原方案通过 lexical 开关保持原两种恢复工具、原片段头部和归档入口；新方案增加 exact JSON lookup、完整小对象、字段路径及文本命中中心片段。旧前缀与工具定义在各组任务内固定，每个任务/组有独立首部标识；不能强制清空服务端缓存。','',
        '不读取答案或目标文件状态来定位证据。调用前冻结数据、配置和包源码；结果覆盖全部三十六个任务条件，执行凭据逐项核对实际 handler 日志。','',
        '[指标](metrics.json) · [全部逐任务](per_checkpoint.csv) · [全部逐请求](per_request.csv) · [最终回答](final_answers.csv) · [执行核对](execution_audit.json) · [来源指纹](source_manifest.json) · [调用账本](usage_ledger.json)','',
        '复现：`python -m token_budget_lab.action_experiment --config configs/locator_deepseek.json --plan`。真实执行使用新 --output，最多 216 次请求。接口见 [接入说明](../../docs/INTEGRATION.md)，后续方案见 [改进路线](../../docs/ROADMAP.md)。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return dict(calls=summary['total_api_attempts'],receipts_reconciled=True,comparisons=comparisons,case_groups=groups)


def audit_transforms(rows,tasks,manifest,output):
    """Recompute every check from journaled actual handler results, not gold."""
    data=Path(__file__).resolve().parents[1]/manifest['config']['data']
    episodes={e['id']:e for e in json.loads(data.read_text(encoding='utf-8'))['episodes']}
    audits=[]
    for task in tasks:
        state=task.get('write_transform')
        if state is None:
            continue
        relevant=[r for r in rows if r['episode_id']==task['episode_id'] and r['mode']==task['mode']]
        # Count canonical positions without materializing a new sandbox. Each
        # actual setup read contributes user/assistant/tool/final messages.
        position=2+4*sum('tool' in op for op in episodes[task['episode_id']]['setup'])
        reads={}
        for response in (r for r in relevant if r['event']=='response'):
            if not response.get('message'):
                continue
            position+=1
            controller=next((r for r in relevant if r['event']=='execution_controller' and r['step']==response['step']),{})
            receipts=controller.get('action_receipts',[])
            for call in response['message'].get('tool_calls',[]):
                name=call['function']['name'];args=json.loads(call['function']['arguments'])
                for receipt in receipts:
                    if receipt['tool']==name and receipt['arguments']==args and receipt['handler_returned']:
                        actual=next((a for a in relevant if a['event']=='business_tool' and a['success'] and a['tool']==name and a['arguments']==args
                            and hashlib.sha256(encode(a['result']).encode()).hexdigest()==receipt['result_sha256']),None)
                        if actual and name.startswith('read_'):
                            message=dict(role='tool',tool_call_id=call['id'],content=encode(actual['result']))
                            reads[source_id(position,message)]=dict(receipt=receipt,result=actual['result'],message=message,position=position)
                        break
                position+=1
            if not response['message'].get('tool_calls') and response['status']=='action_pending':
                position+=1  # Host retry notice enters canonical history.
        for event in state['events']:
            sid=event.get('source_id')
            source=reads.get(sid) if sid else None
            if sid and (source is None or source['receipt']['result_sha256']!=event['result_sha256']
                or source['receipt']['tool']!=event['read_tool'] or source['receipt']['arguments']!=event['read_arguments']):
                raise ValueError('transform source does not match actual read receipt/message')
            rule=next(WriteTransform(**r) for r in state['rules'] if r['write_tool']==event['tool'])
            arguments=None
            # Use actual submitted arguments, including a wrong currency or
            # type, rather than reconstructing fixed arguments from the rule.
            for response in (r for r in relevant if r['event']=='response'):
                for call in response.get('message',{}).get('tool_calls',[]):
                    if call['function']['name']!=event['tool']:
                        continue
                    candidate=json.loads(call['function']['arguments'])
                    if candidate.get(rule.value_argument)!=event['attempted_value']:
                        continue
                    g=FreshWriteGuard([rule])
                    if source:
                        g.observe(rule.read_tool,rule.read_arguments,source['result'],source['message'],source['position'],True)
                    try:
                        g.validate(event['tool'],candidate)
                    except WriteValueRejected:
                        pass
                    if g.events[-1]==event:
                        arguments=candidate
                        break
                if arguments is not None:
                    break
            if arguments is None:
                raise ValueError('transform recomputation differs from runtime check')
            audits.append(dict(episode_id=task['episode_id'],mode=task['mode'],**event,
                submitted_arguments=arguments,source_receipt_verified=source is not None,recomputed=True))
    write(output/'write_validation.json',audits)


def transform_report(summary,metrics,groups,output):
    by={r['mode']:r for r in metrics['runs']}
    total=lambda r:r['prompt_tokens']+r['completion_tokens']
    a=by['context_transform'];comparisons=[]
    for base in ('context_structured','full'):
        b=by[base]
        comparisons.append(dict(variant='context_transform',baseline=base,
            task_quality_delta=a['task_success_rate']-b['task_success_rate'],
            total_token_reduction=1-total(a)/total(b),fixed_reference_cost_reduction=1-a['estimated_usd']/b['estimated_usd'],
            p95_seconds_ratio=a['latency']['p95_seconds']/b['latency']['p95_seconds']))
    write(output/'comparisons.json',comparisons)
    n=metrics['case_count']
    def table(name,fn):
        return '| '+name+' | '+' | '.join(fn(by[m]) for m in metrics['arms'])+' |'
    report=['# 本轮写入值变换核验的真实对照','',
        f'v0.10，{n} 个长历史合成情境、三组、同轮随机顺序。复用冻结的诊断任务，含重复模板；不是新的独立自然任务确认。Full 与插件核验组使用相同写入规则；不加核验的压缩插件作为消融对照。全部组保留相同执行合同。','',
        '| 指标 | Full + 核验 | 压缩插件 | 压缩插件 + 核验 |','|---|---:|---:|---:|',
        table('完整任务通过',lambda r:f"{round(r['task_success_rate']*n)}/{n}"),
        table('答案准确率',lambda r:f"{r['answer_accuracy']:.1%}"),
        table('文件状态准确率',lambda r:f"{r['artifact_state_accuracy']:.1%}"),
        table('本轮实际读取',lambda r:f"{r['fresh_read_completed']}/{n}"),
        table('模型调用',lambda r:str(r['model_calls'])),
        table('恢复调用',lambda r:str(r['recovery_tool_calls'])),
        table('通过/拒绝写入核验',lambda r:f"{r['write_transform_accepted']}/{r['write_transform_rejected']}"),
        table('输入输出总 token',lambda r:f"{total(r):,}"),
        table('固定参考费用 USD',lambda r:f"{r['estimated_usd']:.6f}"),
        table('整任务 P95 秒',lambda r:f"{r['latency']['p95_seconds']:.3f}"),'',
        '## 核验内容','',
        '宿主从当前用户要求声明读取参数、字段选择与允许变换，在实际读取 handler 返回后绑定本轮原文。格式化使用 Decimal，保留数值大小；整数偏移使用明确的常量。提交值不符时先拒绝写入，并反馈真实原文重算值，模型可在原六步上限内纠正。宿主不自动写入。','',
        '核验不读取标准答案、目标文件状态或缺失/重复标签。所有条件题仍只声明相同的必需读取；是否唯一由实际文件决定。规则提示仅包含选择与变换，首次请求没有计算出的答案。组间区别同时包含核验提示与拒绝反馈，不能把通过率变化全部归因于某一次拦截。','',
        '## 本轮结论与轨迹','',
        *[p for finding in metrics['reviewed_findings'] for p in (finding,'')],
        f"插件核验组与 Full 的逐任务答案一致率为 {next(r['agreement_with_full'] for r in summary['runs'] if r['mode']=='context_transform'):.1%}。单次、复用的合成诊断不足以证明一般任务准确率近乎无损；宿主只验证配置的数值写入，最终答案仍独立评分。",'',
        '## 完整任务成本与延迟','']
    for c in comparisons:
        change=lambda v:('减少 ' if v>=0 else '增加 ')+f'{abs(v):.1%}'
        report.append(f"- 核验插件相对 {c['baseline']}：总 token {change(c['total_token_reduction'])}，固定参考费用{change(c['fixed_reference_cost_reduction'])}，P95 为其 {c['p95_seconds_ratio']:.2f} 倍。")
    report += ['',f"本轮共 {summary['total_api_attempts']} 次真实请求，未知 usage {summary['unknown_usage_attempts']} 次，固定参考费用合计 ${summary['research_usage']['estimated_usd']:.6f}。使用命中/未命中分项重算，全部失败、恢复、纠正与最终回答均计入；不是账户账单或实时价格。",'',
        '真正发生压缩的任务另见 [case_groups.json](case_groups.json)。禁止写入任务中的模型写调用尝试仍记违规，即使宿主核验拦截、未落盘。没有增加步骤、重置状态或重跑至通过。','',
        '逐项写入核验从实际 handler 日志重新计算，并核对 canonical 来源 ID、读取参数和原文 SHA-256；不会用结果标签验证来源。既有实验的费用继续保留在调用账本，旧质量成绩不与本轮混合。','',
        '## 可核对资料','',
        '[指标](metrics.json) · [全部任务](per_checkpoint.csv) · [全部请求](per_request.csv) · [回答](final_answers.csv) · [执行凭据](execution_audit.json) · [写入核验](write_validation.json) · [源码指纹](source_manifest.json) · [调用账本](usage_ledger.json)','',
        '复现：`python -m token_budget_lab.action_experiment --config configs/transform_deepseek.json --plan`。实际执行使用新的 --output，最多 216 次请求。接口见 [接入说明](../../docs/INTEGRATION.md)。']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return dict(calls=summary['total_api_attempts'],receipts_reconciled=True,write_checks_recomputed=True,comparisons=comparisons,case_groups=groups)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path)
    args=p.parse_args()
    print(json.dumps(export_current(args.run,args.output,args.diagnostic),ensure_ascii=True))
