"""Reproduce offline diagnostics; optionally analyze an existing paid journal."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from token_budget_lab.analysis import pct
from token_budget_lab.rag_experiment import execute, export

LABELS = dict(fixed_small='固定小预算', fixed_medium='固定中预算', fixed_large='固定大预算',
              adaptive='自适应预算', shuffled_adaptive='打乱自适应预算')


def report(destination):
    destination = Path(destination)
    summaries = {s: json.loads((destination/s/'summary.json').read_text(encoding='utf-8'))
                 for s in ('dev','test')}
    lines = ['# 证据需求与自适应输入预算：第一轮实验', '',
             '本轮检验同一检索器下的预算分配。任务质量先于 token 与速度；离线证据诊断不作为回答准确率。', '',
             '## 方法与对照', '',
             '资料库包含全部 48 篇文章，建立 2,258 个带原文偏移的段落块；查询不预先限定到标准答案所在文章。'
             '块长度上限 1,200 字符，长段落窗口重叠 160 字符，BM25 召回前 24 个候选。'
             '按排名装入完整块，重叠比例至少 60% 的块去重，引用标题与分隔符也计入证据预算。', '',
             '固定预算为 1,800 / 3,600 / 7,200 字符。自适应规则读取问题、首候选的 IDF 加权词项覆盖率和前两名分数差；'
             '复杂问句、低覆盖率或分数接近时提高预算，匹配集中时降低预算。阈值在配置中固定，本轮未依据测试结果调参。', '',
             '打乱预算对照保留每组题目的自适应预算多重集合，以种子 20261012 重新分配到问题，检验分配规则的作用。'
             '它匹配名义平均预算，但完整块装入后的字符数和实际 token 不严格相同；单次打乱仍可能受随机种子影响。', '',
             '本轮关闭应用层回答缓存，不添加语义检索或重排器。开发集和测试集各 100 道题、10 篇文章。'
             '两组问题对应的文章互不重叠；所有文章均作为可检索知识库，这是问题划分，不是未知文章外推评估。', '',
             '## 离线证据诊断', '',
             '| 划分 | 方法 | 答案片段保留 | 完整来源段落覆盖 | 平均证据字符 | 相对大预算减少 |',
             '|---|---|---:|---:|---:|---:|']
    for split, summary in summaries.items():
        for r in summary['runs']:
            lines.append(f"| {split} | {LABELS[r['strategy']]} | {pct(r['answer_span_retained'])} | {pct(r['source_paragraph_covered'])} | {r['mean_context_characters']:.1f} | {pct(r['context_character_reduction'])} |")
    for split, summary in summaries.items():
        adaptive = next(r for r in summary['runs'] if r['strategy']=='adaptive')
        lines.append(f"\n{split} 自适应预算分布：`{json.dumps(adaptive['budgets'],ensure_ascii=False)}`；前 24 候选的答案字符串召回率 {pct(adaptive['candidate_answer_span_recall'])}。")
    lines += ['', '答案片段保留检查任一参考字符串是否出现在所选块正文中；字符串出现可能不是回答该题的证据。'
              '完整来源段落覆盖通过原始段落哈希与所选块偏移区间的并集检查，较保守，也不等同于最小必要证据。'
              '标准答案、正确文章 ID 和来源段落哈希仅用于选取和生成结束后的评分。', '',
              '开发集上自适应方案的答案片段保留率与大预算相同，但测试集低 1 个百分点。'
              '测试集打乱预算的片段保留率也为 95%，因此本轮离线证据不足以证明按问题分配优于同平均预算的随机分配。']
    pilot_path = destination/'deepseek_pilot'/'summary.json'
    if pilot_path.exists():
        summary = json.loads(pilot_path.read_text(encoding='utf-8'))
        manifest = json.loads((pilot_path.parent/'manifest.json').read_text(encoding='utf-8'))
        c = manifest['config']
        lines += ['', '## DeepSeek 真实生成试验', '',
                  f"开发集按文章轮转取 {len(manifest['question_ids'])} 道题，每篇 2 道，共 10 篇。"
                  f"模型请求名 `{c['model']}`；`{c['protocol']}` 协议，配置 thinking=`{c['thinking']}`，"
                  f"每次输出上限 {c['max_tokens']}，总调用上限 {c['max_calls']}。每道题内部用固定种子随机排列方法，单次运行。", '',
                  '所有方法采用相同 JSON 短答案提示，要求 answer、citations、abstain。'
                  'EM/F1 使用 SQuAD 英文归一化；空答、无效 JSON、截断与拒答均纳入质量分母，失败调用的已知 usage 仍计入消耗。'
                  '引用校验只检查 ID 存在，尚未验证引用内容是否支持答案。', '',
                  '| 方法 | EM | F1 | 不可用响应 | 总输入 token | 输出 token | 输入减少 | 合计减少 | P50 秒 | P95 秒 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary['runs']:
            lines.append(f"| {LABELS[r['strategy']]} | {pct(r['exact_match'])} | {pct(r['f1'])} | {r['errors']}/{r['requests']} | {r['prompt_tokens']} | {r['completion_tokens']} | {pct(r['input_token_reduction'])} | {pct(r['total_token_reduction'])} | {r['latency']['p50_seconds']:.3f} | {r['latency']['p95_seconds']:.3f} |")
        lines += ['', '### 质量差与可用性', '',
                  '| 方法 | 配对 EM 差（百分点） | 文章簇 bootstrap 95% 区间 | 质量筛查 | 响应状态 |',
                  '|---|---:|---|---|---|']
        for r in summary['runs']:
            p = r['paired_em_difference']
            delta = f"{p['macro_mean']*100:+.1f}" if p else '—'
            interval = f"[{p['ci95'][0]*100:+.1f}, {p['ci95'][1]*100:+.1f}]" if p else '—'
            lines.append(f"| {LABELS[r['strategy']]} | {delta} | {interval} | `{r['quality_screen']}` | `{json.dumps(r['statuses'],ensure_ascii=False)}` |")
        p = summary.get('paired_adaptive_vs_shuffled_em')
        if p:
            lines += ['', f"自适应相对打乱预算的配对 EM 差为 {p['macro_mean']*100:+.1f} 个百分点，"
                       f"文章簇 bootstrap 95% 区间 [{p['ci95'][0]*100:+.1f}, {p['ci95'][1]*100:+.1f}]。"
                       '本轮未观察到预算分配规则相对打乱对照的优势。']
        lines += ['', '允许质量下降为 0；bootstrap 以文章而非题目为簇。20 题只有 10 个文章簇，'
                  '单次自动指标与区间不足以证明质量等价，仍需人工语义复核与更多独立问题。'
                  '不可用响应会使筛查显示 incomplete，表示可用性要求未满足；已得到 usage 的试验消耗仍可完整汇总。', '',
                  '### 输入计量与速度范围', '',
                  'Anthropic 格式的 `input_tokens` 不含缓存读取和缓存写入。本报告的总输入为 '
                  '`input_tokens + cache_read_input_tokens + cache_creation_input_tokens`；Chat 格式的 `prompt_tokens` 已含缓存命中，不重复加。'
                  '原始日志保留，公开数值由原始 usage 重算，执行代码哈希与分析代码哈希分别记录。', '',
                  '| 方法 | 缓存读取 token | usage 未知请求 | 本地检索 P50 毫秒 | 预算与装入 P50 毫秒 |',
                  '|---|---:|---:|---:|---:|']
        for r in summary['runs']:
            lines.append(f"| {LABELS[r['strategy']]} | {r['provider_cache_hit_tokens']} | {r['unknown_usage_requests']} | {r['retrieval_latency']['p50_seconds']*1000:.3f} | {r['selection_latency']['p50_seconds']*1000:.3f} |")
        lines += ['', '服务商前缀缓存不可控，会影响账单与速度；输入 token 数量减少不等于同百分比费用下降。'
                  '端到端计时包含检索、预算分配、证据装入、网络调用及响应解析，排除建索引、计划预计算、日志 fsync 与参考答案评分。'
                  '未测首 token 延迟、流式响应、并发吞吐量或冷启动总耗时。单轮 P50/P95 不作为稳定速度优势的证明。', '',
                  '字段口径依据 [Anthropic 缓存计量说明](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#tracking-cache-performance)，'
                  '参数依据 [DeepSeek 思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)。']
        if (destination/'FAILURE_CASES.md').exists():
            lines += ['', '逐题复核见 [失败与指标差异案例](FAILURE_CASES.md)。']
    lines += ['', '## 本轮判断与后续验证', '',
              '规则实现了按问题分配预算，并降低了相对固定大预算的平均证据长度。'
              '预算信号仍是词面启发式，不能确认模型是否获得充分证据；第一轮不能直接宣称在准确率不下降时节省 token 或稳定提速。'
              '后续只在开发集改进规则并冻结参数，再进行独立真实生成评测；可加入证据重排、按分面补齐及多种打乱种子，明确区分检索和预算模块的贡献。', '',
              '## 复现与文件', '',
              '离线复现无需 API、第三方包或模型下载：', '',
              '```bash', 'python scripts/reproduce_rag_budget.py --output local_results/rag-budget-reproduction', '```', '',
              '真实生成使用本机 `DEEPSEEK_API_KEY`，先查看调用计划：', '',
              '```bash', 'python -m token_budget_lab.rag_experiment --config configs/rag_budget_deepseek_pilot.json --output local_results/rag-budget-live --plan',
              'python -m token_budget_lab.rag_experiment --config configs/rag_budget_deepseek_pilot.json --output local_results/rag-budget-live --export results/adaptive_budget/deepseek_pilot', '```', '',
              '付费请求先写入 start 记录再调用。已完成的请求可在配置、数据、代码均未变化时用 `--resume` 跳过；'
              '未结算尝试或 API 错误阻止自动重试。修改代码后应使用新输出目录。'
              '现有付费日志的离线重算不产生 API 调用：', '',
              '```bash', 'python scripts/reproduce_rag_budget.py --output local_results/rag-budget-reanalysis --pilot-run local_results/rag-budget-live --publish results/adaptive_budget', '```', '',
              '`dev/`、`test/`、`deepseek_pilot/` 分别提供 manifest、summary 和逐题数值 CSV。'
              '原始模型输出保存在被 Git 忽略的 local_results 中。'
              '文章与题目来自 [SQuAD v1.1 的本项目划分](../../data/PUBLIC_DATA.md)，并非官方 SQuAD 排名。'
              '本轮从整库检索，旧实验给定正确文档，两者任务条件不同，不能把百分比直接比较。', '']
    (destination/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    colors = dict(fixed_small='#8b9baa', fixed_medium='#5377a0', fixed_large='#23384d',
                  adaptive='#158875', shuffled_adaptive='#c28425')
    names = dict(fixed_small='Fixed 1.8k', fixed_medium='Fixed 3.6k', fixed_large='Fixed 7.2k',
                 adaptive='Adaptive', shuffled_adaptive='Shuffled')
    markers = dict(fixed_small='o', fixed_medium='o', fixed_large='o', adaptive='D', shuffled_adaptive='x')
    test = json.loads((destination/'test'/'summary.json').read_text(encoding='utf-8'))
    for r in test['runs']:
        axes[0].scatter(r['mean_context_characters'], r['answer_span_retained']*100,
                        color=colors[r['strategy']], marker=markers[r['strategy']], s=65, label=names[r['strategy']])
    axes[0].set(title='Test: retrieval diagnostic (100 questions)', xlabel='Mean evidence characters',
                ylabel='Answer string retained (%)', ylim=(80,100))
    pilot_path = destination/'deepseek_pilot'/'summary.json'
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
        for r in pilot['runs']:
            axes[1].scatter(r['prompt_tokens']/r['requests'], r['exact_match']*100,
                            color=colors[r['strategy']], marker=markers[r['strategy']], s=65)
        axes[1].set(title='DeepSeek pilot (20 questions; one run)', xlabel='Mean total input tokens (cache included)',
                    ylabel='Short-answer EM (%)', ylim=(50,80))
        labels = [names[r['strategy']] for r in pilot['runs']]
        axes[2].bar(labels, [r['latency']['p50_seconds'] for r in pilot['runs']],
                    color=[colors[r['strategy']] for r in pilot['runs']])
        axes[2].set(title='Pilot latency: P50 (single run)', ylabel='End-to-end seconds')
        axes[2].tick_params(axis='x', rotation=35)
    else:
        for axis in axes[1:]:
            axis.set_visible(False)
    axes[0].legend(loc='lower right', fontsize=8)
    for axis in axes:
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    fig.savefig(destination/'comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', type=Path)
    parser.add_argument('--pilot-run', type=Path, help='Analyze an existing log; never starts API calls')
    parser.add_argument('--plot', action='store_true', help='Optional matplotlib publication figure')
    args = parser.parse_args()
    destination = args.publish or args.output/'publication'
    for split in ('dev','test'):
        config = json.loads((ROOT/f'configs/rag_budget_{split}.json').read_text(encoding='utf-8'))
        execute(config, args.output/split)
        export(args.output/split, destination/split)
    if args.pilot_run:
        export(args.pilot_run, destination/'deepseek_pilot')
    report(destination)
    if args.plot:
        plot(destination)
    print(f'Offline diagnostics and report: {destination.resolve()}')


if __name__ == '__main__':
    main()
