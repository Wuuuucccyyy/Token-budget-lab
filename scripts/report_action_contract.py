"""Publish all outcomes and reconcile host receipts with actual business calls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_budget_lab.memory import encode
from token_budget_lab.tool_experiment import export


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    export(args.run, dest/'deepseek_test')
    summary = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    raw = (args.run/'requests.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    tasks = [r for r in rows if r['event']=='task_result']
    failures, audits = [], []
    for task in tasks:
        chain = [r for r in rows if r.get('episode_id')==task['episode_id'] and r.get('mode')==task['mode']]
        business = [r for r in chain if r['event']=='business_tool']
        policy = task['execution_contract']
        verified = None
        if policy:
            receipts = policy['receipts']
            verified = len(receipts)==len(business) and all(
                r['tool']==b['tool'] and r['arguments']==b['arguments']
                and r['handler_returned']==b['success']
                and (not b['success'] or r['result_sha256']==hashlib.sha256(encode(b['result']).encode()).hexdigest())
                for r,b in zip(receipts, business))
            if not verified:
                raise ValueError('execution receipts do not reconcile with actual business handlers')
        audits.append(dict(episode_id=task['episode_id'], mode=task['mode'],
                           fresh_read_completed=task['fresh_read_completed'], status=task['status'],
                           task_success=task['task_success'], execution_contract=policy,
                           receipts_match_business_calls=verified))
        if not task['task_success']:
            failures.append(dict(result=task, model_responses=[r for r in chain if r['event']=='response'],
                                 business_actions=business, controller_events=[r for r in chain if r['event']=='execution_controller']))
    write(dest/'execution_audit.json', audits)
    write(dest/'failure_cases.json', failures)
    write(dest/'controller_events.json', [r for r in rows if r['event']=='execution_controller'])
    write(dest/'cost_accounting.json', dict(rates=summary['manifest']['config']['usd_per_million'],
        pricing_source=summary['manifest']['pricing_source'], interpretation=summary['manifest']['pricing_interpretation'],
        runs=summary['runs'], total_attempts=summary['total_api_attempts'],
        unknown_usage_attempts=summary['unknown_usage_attempts'], research_usage=summary['research_usage']))
    sources = {}
    for source in (args.run/'execution_sources').glob('*.py'):
        content = source.read_bytes()
        sources[source.name] = dict(sha256=hashlib.sha256(content).hexdigest(), text=content.decode('utf-8'))
    current = Path(__file__).resolve().parents[1]/'token_budget_lab'
    differences = [name for name, source in sources.items()
                   if hashlib.sha256((current/name).read_bytes()).hexdigest()!=source['sha256']]
    write(dest/'source_snapshot.json', dict(code_sha256=summary['manifest']['code_sha256'], files=sources,
        current_package_differences=differences,
        note='exact package source bytes captured before first paid request; after run, ordered identical-tool obligations and their stable-read bypass were fixed; pilot declares different read/write/check tool names with one required read'))
    write(dest/'usage_ledger.json', dict(journal_sha256=hashlib.sha256(raw).hexdigest(),
        total_attempts=summary['total_api_attempts'], known_responses=sum(r['event']=='response' and r.get('usage') is not None for r in rows),
        unknown_usage_attempts=summary['unknown_usage_attempts'], max_steps=6, paid_call_cap=144,
        includes_failures=True, automatic_retries=False, research_cost=summary['research_usage'], actual_invoice=None))
    by = {r['mode']:r for r in summary['runs']}
    total = lambda r:r['prompt_tokens']+r['completion_tokens']
    comparisons = [dict(variant=a, baseline=b, task_quality_delta=by[a]['task_success_rate']-by[b]['task_success_rate'],
                        total_token_reduction=1-total(by[a])/total(by[b]),
                        fixed_price_cost_reduction=1-by[a]['estimated_usd']/by[b]['estimated_usd'],
                        p95_seconds_ratio=by[a]['latency']['p95_seconds']/by[b]['latency']['p95_seconds'])
                   for a,b in [('context_contract','full_contract'), ('context_contract','context')]]
    write(dest/'comparisons.json', comparisons)
    def table(title, formatter):
        return '| '+title+' | '+' | '.join(formatter(r) for r in summary['runs'])+' |'
    lines = ['# 本轮读取与必需操作执行检查', '',
        'v0.7，2026-10-12。六个长历史合成诊断、四组真实 DeepSeek 对照。两条是 v0.6 日志任务复查，四条在历史读取之后修改当前文件。单任务最多六次模型请求，上限保持不变。', '',
        '## 固定实现结果', '',
        '| 指标 | 原生 Full | Full + 执行检查 | 插件 | 插件 + 执行检查 |',
        '|---|---:|---:|---:|---:|',
        table('正确完成任务', lambda r:f"{round(r['task_success_rate']*6)}/6"),
        table('指定本轮读取完成', lambda r:f"{r['fresh_read_completed']}/6"),
        table('答案准确率', lambda r:f"{r['answer_accuracy']:.1%}"),
        table('文件状态准确率', lambda r:f"{r['artifact_state_accuracy']:.1%}"),
        table('模型调用', lambda r:str(r['model_calls'])),
        table('输入输出总 token', lambda r:f"{total(r):,}"),
        table('缓存未命中输入 token', lambda r:f"{r['cache_miss_tokens']:,}"),
        table('固定非高峰价重估 USD', lambda r:f"${r['estimated_usd']:.6f}"),
        table('整任务 P95 秒', lambda r:f"{r['latency']['p95_seconds']:.3f}"), '',
        '插件组实际发生压缩的任务为 6/6，成功率与全组相同。计入恢复、未完成操作和失败的全部调用；没有把答案正确但漏操作的任务记为成功。', '',
        '## 结论与剩余问题', '',
        '未启用执行检查的插件漏读一条任务：答案、报告和检查结果正确，仍未调用用户指定的 read_log。启用后六条指定读取均实际执行，完整任务从 3/6 提升到 5/6。这个小样本结果支持执行检查的有效性，不证明普遍准确率。', '',
        '启用执行检查的插件仍有一条日志恢复循环失败：已经实际读取 scan_d，但搜索片段截在命中事件之前，模型反复搜索并分页，恢复预算耗尽；报告和检查未完成。完整任务仍低于同样启用执行检查的 Full 6/6。执行合同没有把这个失败改成成功。', '',
        '执行检查确保已声明操作没有完成时不会接受最终答案；不能保证模型完成任务或写入值使用正确来源。下一项需要改进的是命中事件的完整片段与精确字段定位，不能将它混同为漏读问题。', '']
    comparison = comparisons[0]
    lines += [f"相对 Full + 执行检查，插件 + 执行检查总 token 减少 {comparison['total_token_reduction']:.1%}，固定价格费用减少 {comparison['fixed_price_cost_reduction']:.1%}，P95 是其 {comparison['p95_seconds_ratio']:.2f} 倍。质量仍有差距，因此不能宣称近乎无损且更快。", '',
        '## 需求与评分边界', '',
        'execution contract 使用宿主声明的本轮用户意图，包括读取工具、文件参数、写入依赖和检查成功条件。它不含答案、目标金额或目标配置数值，也不读取评分标签；并非通用自然语言自动提取。四组使用相同用户任务与实际工具，两组执行检查使用相同合同。', '',
        'setup 通过真实文件工具生成旧历史；live_updates 在模型调用前更新沙盒文件，原始历史保留旧值。两条旧任务是诊断复查，四条新任务沿用现有模板，没有独立自然任务确认。禁止使用结果当作一般任务准确率或版本间正式非劣性证据。', '',
        '最终评分同时检查答案格式及值、文件状态、实际业务工具、指定读取参数和禁止写入条件。执行凭据与实际 handler 日志逐条核对，结果见 [execution_audit.json](execution_audit.json)。', '',
        '## 成本与复现', '',
        f"共 {summary['total_api_attempts']} 次真实调用，未知 usage {summary['unknown_usage_attempts']} 次，无额外缓存重放，无自动重试。按 v0.6 相同固定非高峰价格重估约 ${summary['research_usage']['estimated_usd']:.6f}，不是到账账单；高峰价格不同。分项见 [费用核算](cost_accounting.json)。", '',
        '每个任务/组使用独立首部标识，任务内追加工具消息并保留固定合同与工具定义。无法清空服务端缓存，使用实际 usage 分项计量。', '',
        '调用前的源码完整保存在 source_snapshot.json。运行后补充修复同工具、同参数的两次顺序读取匹配及相应的稳定读取回显绕过；本轮只有一次必需读取，读取、写入和检查均为不同工具，不触发该分支，未因这个修改重发付费请求。额外分支以单元测试验证，不计入真实模型成绩。', '',
        f"保留全部 {len(failures)} 条失败，见 [失败轨迹](failure_cases.json)。未增加步数、替模型执行业务操作或修改错误状态。", '',
        '公开 [调用前源码快照](source_snapshot.json)、[逐请求](deepseek_test/per_request.csv)、[逐任务](deepseek_test/per_episode.csv)、[实际工具动作](deepseek_test/business_actions.csv)、[最终回答](deepseek_test/final_answers.csv) 和 [调用账本](usage_ledger.json)。', '',
        '```bash',
        'python -m token_budget_lab.action_experiment --config configs/action_contract_deepseek.json --plan',
        '```', '',
        '真实运行需要本机 DEEPSEEK_API_KEY 与新的 --output 目录，最多 144 次尝试。接入接口见 [REQUIRED_ACTIONS.md](../../docs/REQUIRED_ACTIONS.md)。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(dest), failures=len(failures), receipts_reconciled=True, comparisons=comparisons), ensure_ascii=True))


if __name__=='__main__':
    main()
