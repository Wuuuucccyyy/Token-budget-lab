"""Regenerate the long-context report, usage ledger and optional comparison plot."""
import argparse
import csv
import json
from pathlib import Path


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def seconds(value):
    return '—' if value is None else f'{value:.3f}'


def ledger_entry(path, label):
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    starts=[r for r in rows if r['event']=='start']
    responses=[r for r in rows if r['event']=='response']
    usage=[r['usage'] for r in responses if r.get('usage') is not None]
    return dict(label=label,attempts=len(starts),responses=len(responses),unknown_usage_requests=len(starts)-len(usage),
                known_prompt_tokens=sum(u['prompt_tokens'] for u in usage),
                known_completion_tokens=sum(u['completion_tokens'] for u in usage))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    summary=json.loads((args.primary/'summary.json').read_text(encoding='utf-8'))
    runs={r['mode']:r for r in summary['runs']}
    full,recover=runs['full'],runs['recover']
    m=summary['manifest']
    entries=[ledger_entry(args.diagnostic/'requests.jsonl','interrupted_diagnostic'),
             ledger_entry(args.primary/'requests.jsonl','fixed_implementation_run')]
    ledger=dict(runs=entries,total_attempts=sum(r['attempts'] for r in entries),
                unknown_usage_requests=sum(r['unknown_usage_requests'] for r in entries),
                known_prompt_tokens=sum(r['known_prompt_tokens'] for r in entries),
                known_completion_tokens=sum(r['known_completion_tokens'] for r in entries),
                interpretation='known usage is a lower bound when requests have unknown usage; not a monetary bill')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'usage_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
    lines=['# 长历史证据选择与连续工具实验','',
        '2026-10-11，v0.4，DeepSeek 真实生成。主要问题是：当历史增长、当前任务只需要其中部分证据时，按事实和时间选择原文能否减少整个任务的 token，同时保持任务正确性。','',
        '## 设置','',
        f'- 本轮包含 {len(m["episode_ids"])} 个长度条件任务、{m["base_scenarios"]} 个基础场景，两组各 {m["checkpoints"]} 个连续检查点。各类型场景来自同一合成模板，不是独立自然任务。',
        f'- 长度分布为 `{m["context_counts"]}`；相同场景的长度变体不作为新的独立场景。',
        '- 原生 Full 不使用插件、归档和恢复工具；恢复组按主体、字段和日期选原始整轮。当前工具链及全部用户要求保留。Full 请求关闭恢复工具。',
        '- 统一业务工具、JSON 输出、关闭思考模式，模型 `deepseek-flash`，每次最多输出 1,024 token。方法顺序按固定种子随机化。',
        '- 配置是 9,000 个序列化字符、当前整轮窗口 1、预计至少减少 10% 字符。字符比例不冒充 API token 节省。',
        '- 历史由真实本地文件读取的脚本回放生成；检查点由模型自主执行。模型需要计算、写文件和运行 Python 检查，不能只给正确答案。','',
        '## 整体结果','',
        '| 指标 | 原生 Full | 恢复插件 |','|---|---:|---:|',
        f'| 检查点通过率 | {pct(full["task_success_rate"])} | {pct(recover["task_success_rate"])} |',
        f'| 完整任务通过率 | {pct(full["episode_success_rate"])} | {pct(recover["episode_success_rate"])} |',
        f'| 答案准确率 | {pct(full["answer_accuracy"])} | {pct(recover["answer_accuracy"])} |',
        f'| 文件状态准确率 | {pct(full["artifact_state_accuracy"])} | {pct(recover["artifact_state_accuracy"])} |',
        f'| 模型请求 | {full["model_calls"]} | {recover["model_calls"]} |',
        f'| 恢复工具函数请求（含拒绝） | {full["recovery_tool_calls"]} | {recover["recovery_tool_calls"]} |',
        f'| 输入 token | {full["prompt_tokens"]:,} | {recover["prompt_tokens"]:,} |',
        f'| 输出 token | {full["completion_tokens"]:,} | {recover["completion_tokens"]:,} |',
        f'| 输入输出合计 | {full["prompt_tokens"]+full["completion_tokens"]:,} | {recover["prompt_tokens"]+recover["completion_tokens"]:,} |',
        f'| 未知 usage 请求 | {full["unknown_usage_requests"]} | {recover["unknown_usage_requests"]} |',
        f'| 检查点 P50 秒 | {seconds(full["latency"]["p50_seconds"])} | {seconds(recover["latency"]["p50_seconds"])} |',
        f'| 检查点 P95 秒 | {seconds(full["latency"]["p95_seconds"])} | {seconds(recover["latency"]["p95_seconds"])} |','',
        f'计入所有请求后，总 token 减少 {pct(recover["total_token_reduction"])}。这个比例是模型输入输出用量，不是账单降幅。',
        f'恢复组请求模式为 `{recover["request_history_modes"]}`；回退及保留原因见逐请求 CSV。','',
        '## 长度条件','',
        '| 条件 | 每组任务数 | 初始历史字符均值 | Full 检查点通过 | 插件检查点通过 | Full 完整任务通过 | 插件完整任务通过 | 总 token 减少 | 压缩请求/插件请求 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    contexts={}
    for context in ('short','medium','long'):
        f=next(r for r in summary['by_context'] if r['mode']=='full' and r['context_length']==context)
        r=next(r for r in summary['by_context'] if r['mode']=='recover' and r['context_length']==context)
        contexts[context]=(f,r)
        lines.append(f'| {context} | {r["episodes"]} | {f["initial_characters"]:,.0f} | {pct(f["checkpoint_success_rate"])} | {pct(r["checkpoint_success_rate"])} | {pct(f["episode_success_rate"])} | {pct(r["episode_success_rate"])} | {pct(r["total_token_reduction"])} | {r["compressed_requests"]}/{r["requests"]} |')
    lines+=['','同一条件下两组比较相同任务。short 样本数较少；不同条件的原始 token 总量不直接用于比较效率。图中成本按完整任务平均。',
        'short 未压缩、未注册恢复工具，四个场景第一次请求的历史哈希一致。两组通过率的差异不能归因于压缩收益，单次生成及后续轨迹有波动。','',
        '## 实际发生压缩的任务','']
    for r in summary['compressed_subset']:
        lines += [f'- 至少一次压缩的任务：{r["episodes"]} 个，来自 {r["base_scenarios"]} 个基础场景。',
            f'- 这些任务的完整任务通过率：插件 {pct(r["episode_success_rate"])}，相同任务的 Full {pct(r["matching_full_episode_success_rate"])}。',
            f'- 这些任务的全部检查点通过率：插件 {pct(r["compressed_episode_checkpoint_success_rate"])}，相同检查点的 Full {pct(r["matching_full_checkpoint_success_rate"])}。',
            f'- 实际压缩的检查点：{r["actually_compressed_checkpoints"]} 个，通过率 {pct(r["compressed_checkpoint_success_rate"])}。',
            f'- 这些任务计入恢复及回退的全部请求后，总 token 减少 {pct(r["total_token_reduction"])}。',
            f'- 配对质量差按 base_id 聚类：`{r["paired_quality_difference"]}`。']
    lines+=['','压缩子集是事后诊断，包含失败任务；不能通过只看此子集证明总体质量不下降。','',
        '## 中断的诊断调用','',
        f'在固定实现前发现本轮写入结果会替换原问题的历史来源，停止了 {entries[0]["attempts"]} 次尝试的诊断运行。已知 usage 和完成的失败任务保存在 `interrupted_diagnostic/`，没有删掉或并入固定实现的主对照。',
        f'两批合计 {ledger["total_attempts"]} 次尝试；未知 usage 为 {ledger["unknown_usage_requests"]} 次。`usage_ledger.json` 计入诊断与主对照所有已知输入输出，不把未知请求当成零成本。',
        '主对照使用诊断后固定的实现；部分同批测试场景已在诊断中被观察，因此这是一轮原型诊断结果，不是独立确认实验。','',
        '## 失败与限制','',
        '- 长历史增加可删除的信息量，但复杂问题仍可能导致错误选择、漏执行工具及后续状态错误。三个连续阶段不自动修复，错误会传递。',
        '- 模型可能重复请求恢复，或在恢复工具关闭后仍生成未注册函数请求。宿主拒绝未注册工具；调用耗尽六步上限时任务失败。逐步轨迹见 [失败案例](FAILURE_CASES.md)。',
        '- 结构化选择依赖明确主体、字段和日期；任意自然语言、隐含指代、多步隐藏依赖没有得到充分验证。',
        '- 9,000 字符上限较保守。受保护的当前工具输出过大时会回退，即使旧历史可压缩，也可能没有净收益。',
        '- 一个种子、一次运行、有限模板；按基础场景聚类的区间不意味着已证明近乎无损。没有生产流量或官方 LongMemEval 成绩。',
        '- 服务商缓存和网络影响速度与价格，未测并发及首 token 延迟。完整任务延迟包括历史准备、所有模型调用和业务工具，排除初始回放及评分。','',
        '## 审计与复现','',
        '- `deepseek_test/per_request.csv`：实际请求模式、选择原因、字段与日期需求、恢复工具开关及 usage。',
        '- `deepseek_test/per_checkpoint.csv`：答案、文件状态、操作完成、约束与累计请求数。',
        '- `deepseek_test/per_episode.csv`：所有任务的累计 token、实际压缩请求、回退请求与完整任务通过。',
        '- `deepseek_test/business_actions.csv`、`artifact_states.json`、`final_answers.csv`：业务执行、最终文件状态及原始回答。',
        '- `offline_dev/summary.json`：脚本驱动检查，API token 和大模型准确率为空。',
        '- [协议与运行命令](../../docs/LONG_CONTEXT_EVALUATION.md)。','']
    with (args.primary/'requests.jsonl').open(encoding='utf-8') as handle:
        failures=[r for line in handle if (r:=json.loads(line))['event']=='task_result' and not r['task_success']]
    if failures:
        lines+=['### 未通过检查点','', '| 任务 | 条件 | 阶段 | 实际答案 | 答案正确 | 状态正确 | 必需工具完成 |','|---|---|---:|---|---|---|---|']
        for r in failures:
            answer=r['answer'].replace('|','/').replace('\n',' ')[:100]
            lines.append(f'| {r["episode_id"]} | {r["mode"]} | {r["checkpoint"]} | {answer or "空"} | {r["answer_correct"]} | {r["state_correct"]} | {r["required_tools_completed"]} |')
        lines+=['']
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(1,3,figsize=(14,4.5))
        xs=np.arange(3)
        for index,(mode,color) in enumerate([('full','#475569'),('recover','#0f766e')]):
            rows=[contexts[c][index] for c in ('short','medium','long')]
            offset=(index-.5)*.34
            quality=[100*r['episode_success_rate'] for r in rows]
            costs=[r['total_tokens']/r['episodes'] for r in rows]
            p50=[r['episode_latency']['p50_seconds'] for r in rows]
            p95=[r['episode_latency']['p95_seconds'] for r in rows]
            axes[0].bar(xs+offset,quality,.32,label=mode,color=color)
            axes[1].bar(xs+offset,costs,.32,label=mode,color=color)
            axes[2].bar(xs+offset,p50,.32,label=mode,color=color)
            axes[2].errorbar(xs+offset,p50,yerr=[[0]*3,[b-a for a,b in zip(p50,p95)]],fmt='none',ecolor=color,capsize=4)
            for x,y,r in zip(xs+offset,quality,rows):
                axes[0].text(x,y+2,f'{round(r["episode_success_rate"]*r["episodes"])}/{r["episodes"]}',ha='center',fontsize=9)
        for ax in axes:
            ax.set_xticks(xs,['short','medium','long']);ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        axes[0].set(title='Whole-episode success',ylabel='Percent',ylim=(0,115))
        axes[1].set(title='All calls: tokens per episode',ylabel='Input + output tokens')
        axes[2].set(title='Episode latency: P50 / P95',ylabel='Seconds')
        axes[0].legend(frameon=False)
        fig.suptitle('DeepSeek: 8 base scenarios, matched length conditions; one synthetic diagnostic run',fontsize=11)
        fig.tight_layout()
        fig.savefig(args.output/'comparison.png',dpi=170)
        plt.close(fig)
        lines+=['![完整任务质量、累计成本及速度](comparison.png)','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(ledger,indent=2))


if __name__=='__main__':main()
