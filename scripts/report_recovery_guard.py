"""Regenerate recovery-loop diagnostics from both fully metered executions."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def journal(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def count(rows):
    responses = [r for r in rows if r['event'] == 'response']
    calls = [(r, c['function']['name']) for r in responses for c in r.get('message', {}).get('tool_calls', [])]
    return dict(recovery_requests=sum(n.startswith('history_') for r, n in calls),
        disabled_recovery_requests=sum(n.startswith('history_') and not r['recovery_tools_enabled'] for r, n in calls),
        suppressed_read_requests=sum(n in r.get('suppressed_snapshot_reads', []) for r, n in calls),
        request_guard_reasons=dict(Counter(r.get('recovery_guard_reason') for r in responses
                                          if r.get('recovery_guard_reason'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, records, ledger = {}, {}, []
    for label, folder in [('diagnostic', args.diagnostic), ('fixed_implementation', args.primary)]:
        summaries[label] = read(folder/'summary.json')
        records[label] = journal(folder/'requests.jsonl')
        starts = [r for r in records[label] if r['event'] == 'start']
        responses = [r for r in records[label] if r['event'] == 'response']
        usage = [r['usage'] for r in responses if r.get('usage') is not None]
        ledger.append(dict(label=label, attempts=len(starts), responses=len(responses),
            unknown_usage_requests=len(starts)-len(usage),
            prompt_tokens=sum(u['prompt_tokens'] for u in usage),
            completion_tokens=sum(u['completion_tokens'] for u in usage)))
    save(args.output/'usage_ledger.json', dict(runs=ledger,
        total_attempts=sum(r['attempts'] for r in ledger),
        unknown_usage_requests=sum(r['unknown_usage_requests'] for r in ledger),
        prompt_tokens=sum(r['prompt_tokens'] for r in ledger),
        completion_tokens=sum(r['completion_tokens'] for r in ledger),
        scope='both executions in this loop-control study; earlier studies remain separate'))
    audit = {}
    for label, rows in records.items():
        audit[label] = {mode: count([r for r in rows if r['mode'] == mode]) for mode in ('full', 'recover')}
    save(args.output/'guard_summary.json', audit)
    save(args.output/'guard_events.json', {label: [r for r in rows if r['event'] == 'recovery_guard'
            and (r['memory_tool_events'] or r['next_recovery_guard_reason'])] for label, rows in records.items()})
    primary = summaries['fixed_implementation']
    full, recover = [next(r for r in primary['runs'] if r['mode'] == mode) for mode in ('full', 'recover')]
    pct = lambda n: f'{n:.1%}'
    lines = ['# 恢复和工具循环控制实验', '',
        'v0.5，DeepSeek 真实调用。目标是限制没有增加证据的恢复，以及重复读取当前文件导致的步骤耗尽。任务正确性先于 token 和速度。', '',
        '## 实现', '',
        '- 搜索标记已可见记录，排除当前任务轮次，预览原始工具内容。恢复返回新增、已可见和已请求 ID；重复恢复不再次添加原文。',
        '- 同一任务最多使用 2 个模型响应轮次进行恢复；发生恢复后，为业务操作及最终回答保留 3 次模型调用。相同搜索、无新增证据或非法恢复请求触发一次全量回退，并锁定至本题结束。',
        '- 全量请求加入发送视图状态通知并移除历史工具。再次请求关闭的工具返回完整历史已可见的明确反馈；连续两轮仍只请求关闭的工具，返回失败状态，不填造最终答案。',
        '- 固定沙盒显式声明 read_config / read_orders / read_report 为无参数、任务内稳定的只读操作。读过后暂时移除，其他成功业务操作后失效并重新开放。参数化读取、外部异步变化和写操作不适用此缓存。',
        '- 所有原始指令、用户请求、工具结果和完整调用链保留。控制通知不写入规范历史，不包含参考值；模型仍负责计算、写入和检查。', '',
        '## 设置与诊断边界', '',
        '- 6 个长历史任务，两组各 18 个连续检查点；9,000 字符预算，窗口 1，预计至少减少 10% 字符才压缩。',
        '- 两组均使用相同业务工具、deepseek-flash、关闭思考、JSON Output、每次最多 1,024 输出 token，每检查点最多 6 次调用。Full 为无插件原生历史。',
        '- 原先已观察 temporal-03、temporal-05 和 tool_execution-04；另三个基础场景在第一轮首次观察。两次运行任务相同，固定实现复查中已没有未观察场景。',
        '- 第一轮仅控制恢复，发现稳定读取仍会循环；第二轮固定完整实现后复查。两轮都保留，不能挑选其中较好的任务作为主结果，也不能把后一次称为独立质量确认。',
        '- 每轮最多 180 次 API 请求，没有自动重试。方法顺序按固定种子打乱，文件状态和错误跨检查点传播，不修复中间文件。', '',
        '## 固定实现结果', '',
        '| 指标 | 原生 Full | 插件 |', '|---|---:|---:|',
        f'| 检查点通过率 | {pct(full["task_success_rate"])} | {pct(recover["task_success_rate"])} |',
        f'| 完整任务通过率 | {pct(full["episode_success_rate"])} | {pct(recover["episode_success_rate"])} |',
        f'| 答案准确率 | {pct(full["answer_accuracy"])} | {pct(recover["answer_accuracy"])} |',
        f'| 文件状态准确率 | {pct(full["artifact_state_accuracy"])} | {pct(recover["artifact_state_accuracy"])} |',
        f'| 必需工具完成率 | {pct(full["required_tool_completion"])} | {pct(recover["required_tool_completion"])} |',
        f'| 模型调用 | {full["model_calls"]} | {recover["model_calls"]} |',
        f'| 输入 token | {full["prompt_tokens"]:,} | {recover["prompt_tokens"]:,} |',
        f'| 输出 token | {full["completion_tokens"]:,} | {recover["completion_tokens"]:,} |',
        f'| 输入输出合计 | {full["prompt_tokens"]+full["completion_tokens"]:,} | {recover["prompt_tokens"]+recover["completion_tokens"]:,} |',
        f'| 检查点 P50 秒 | {full["latency"]["p50_seconds"]:.3f} | {recover["latency"]["p50_seconds"]:.3f} |',
        f'| 检查点 P95 秒 | {full["latency"]["p95_seconds"]:.3f} | {recover["latency"]["p95_seconds"]:.3f} |', '',
        f'插件整体总 token 变化为减少 {pct(recover["total_token_reduction"])}；负数表示增加。速度按整个检查点计入筛选、模型和实际工具执行，不只统计首次请求。', '',
        f'插件终止状态：`{recover["status_counts"]}`。配对质量差：`{recover["paired_quality_difference"]}`。这组单次、合成且诊断后复查的样本不能证明质量非劣效或普遍净节省。', '',
        '## 两轮均保留', '',
        '| 运行 | Full 通过 | 插件通过 | 插件步骤耗尽 | 插件截断 | 插件恢复请求（含拒绝） | 插件请求关闭的历史工具 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label, summary in summaries.items():
        f, r = [next(x for x in summary['runs'] if x['mode'] == m) for m in ('full', 'recover')]
        a = audit[label]['recover']
        lines.append(f'| {label} | {round(f["task_success_rate"]*f["checkpoints"])}/{f["checkpoints"]} | '
            f'{round(r["task_success_rate"]*r["checkpoints"])}/{r["checkpoints"]} | {r["status_counts"].get("step_limit",0)} | '
            f'{r["status_counts"].get("truncated",0)} | {a["recovery_requests"]} | {a["disabled_recovery_requests"]} |')
    lines += ['', '请求关闭的函数与实际执行是不同事件；所有请求和拒绝仍计入模型成本。恢复轮次上限不代表业务工具一定能在剩余步数内完成，明显停滞会以失败终止。', '',
        '## 实际压缩任务', '',
        '按至少发送一次压缩视图选出任务，再累计该任务全部调用，包括回退、恢复、失败以及后续检查点。未删除失败。以下为运行后选择的诊断子集：', '']
    for subset in primary['compressed_subset']:
        lines += [f'- 任务 {subset["episodes"]}；全部检查点通过率 {pct(subset["compressed_episode_checkpoint_success_rate"])}，'
                  f'匹配 Full {pct(subset["matching_full_checkpoint_success_rate"])}。',
                  f'- 真正发生压缩的检查点 {subset["actually_compressed_checkpoints"]}；通过率 {pct(subset["compressed_checkpoint_success_rate"])}。',
                  f'- 全任务累计 token {subset["total_tokens"]:,}；匹配 Full {subset["matching_full_total_tokens"]:,}；减少 {pct(subset["total_token_reduction"])}。']
    lines += ['', '## 成本与审计', '',
        f'两轮累计 {sum(r["attempts"] for r in ledger)} 次 API 尝试，未知 usage {sum(r["unknown_usage_requests"] for r in ledger)} 次。'
        '主结果表只比较固定实现这一轮；研究总开销另计两轮，不能把诊断调用当成免费。', '',
        '[完整用量账本](usage_ledger.json) · [控制事件](guard_events.json) · [逐请求和文件审计](deepseek_test/) · '
        '[第一轮诊断与源码快照](diagnostic/) · [失败轨迹](FAILURE_CASES.md)', '',
        '固定实现仍未通过独立质量验收。后续需要新自然任务、重复运行及预设质量门槛，不能依据单次通过率接近就默认启用。', '',
        '复现配置为 `configs/agent_recovery_guard_deepseek.json`；运行后可用 '
        '`scripts/export_tool_audit.py` 与 `scripts/report_recovery_guard.py` 重建审计和报告。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    failures, traces = ['# 保留的失败与工具轨迹', '',
        '所有失败均保留。以下消息来自原创合成任务，按实际响应提取；不从解释中重评分。', ''], {}
    for label, rows in records.items():
        bad = [r for r in rows if r['event'] == 'task_result' and not r['task_success']]
        failures += [f'## {label}', '', '| 任务 | 模式 | 检查点 | 状态 | 答案 | 文件 | 必需工具 |', '|---|---|---:|---|---|---|---|']
        for r in bad:
            failures.append(f'| {r["episode_id"]} | {r["mode"]} | {r["checkpoint"]} | {r["status"]} | '
                f'{r["answer_correct"]} | {r["state_correct"]} | {r["required_tools_completed"]} |')
        keys = {(r['episode_id'], r['mode'], r['checkpoint']) for r in bad}
        trace = []
        for r in rows:
            if r['event'] == 'response' and (r['episode_id'], r['mode'], r['checkpoint']) in keys:
                trace.append({**{k:r.get(k) for k in ('episode_id','mode','checkpoint','step','status',
                    'history_mode','history_reason','suppressed_snapshot_reads','usage')},
                    'tool_calls': r.get('message',{}).get('tool_calls',[]),
                    'answer_excerpt': (r.get('message',{}).get('content') or '').strip()[:240]})
        traces[label] = trace
        failures += ['', f'逐步响应见 [failure_trace.json](failure_trace.json) 的 `{label}`，工具操作与最终文件见该轮审计目录。', '']
    save(args.output/'failure_trace.json', traces)
    failures += ['## 固定实现中的剩余错误', '',
        '`temporal-05-long` 的检查点 1 需要截至 2026-10-06 的有效快照值 985，模型却写入首次快照的 615。'
        '目标来源消息已经保留，仍发生版本使用错误。实际 run_checks 通过，但它检查类型与业务约束，不能证明选择了正确历史版本。', '',
        '检查点 2 的 PASS 与实际检查结果一致，但错误的 timeout_ms 继续保留，因此文件状态评分仍失败。'
        'Full 的另一个失败来自拒答题最终格式，不据事后解析改写成绩。', '',
        '后续应优先验证来源日期与本次写入值的对应关系，并控制无关日志读取；循环顺利结束不是任务正确性的充分条件。']
    (args.output/'FAILURE_CASES.md').write_text('\n'.join(failures)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
