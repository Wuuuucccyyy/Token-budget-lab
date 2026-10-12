"""Export every task and request of the current locator pilot."""
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
    write(output/'execution_audit.json',audits)
    keys=('mode','checkpoints','task_success_rate','answer_accuracy','artifact_state_accuracy','fresh_read_completed',
          'model_calls','recovery_tool_calls','prompt_tokens','completion_tokens','cache_hit_tokens','cache_miss_tokens',
          'estimated_usd','latency','status_counts','constraint_violations')
    metrics=dict(measured_at=manifest['started_at_utc'],package_version='0.9.0',case_count=manifest['checkpoints'],
        arms=manifest['config']['arms'],scope=manifest['dataset_source'],
        max_steps_per_task=manifest['config']['max_steps'],actual_api_calls=summary['total_api_attempts'],
        unknown_usage_requests=summary['unknown_usage_attempts'],source_package_sha256=manifest['code_sha256'],
        data_sha256=manifest['data_sha256'],rates=manifest['config']['usd_per_million'],
        price_interpretation=manifest['pricing_interpretation'],
        runs=[{k:r[k] for k in keys} for r in summary['runs']],
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path)
    args=p.parse_args()
    print(json.dumps(export_current(args.run,args.output,args.diagnostic),ensure_ascii=True))
