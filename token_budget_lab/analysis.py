"""Aggregate experiments, cluster bootstrap, and regenerate publication artifacts."""
from __future__ import annotations
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lower = int(index)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def interval(values):
    return [quantile(values, .025), quantile(values, .975)]


def latency_stats(rows, field='end_to_end_seconds'):
    values = [r[field] for r in rows if r.get(field) is not None]
    return dict(samples=len(values), mean_seconds=mean(values),
                p50_seconds=quantile(values, .5), p95_seconds=quantile(values, .95))


def quality_screen(backend, metric, paired, expected, observed, errors, allowed_drop=0, baseline_score=None):
    """Quality precedes efficiency. A proxy never authorizes a task-quality claim."""
    if backend != 'deepseek' or metric != 'exact_match':
        return 'proxy_only'
    if errors or observed != expected or paired is None:
        return 'incomplete'
    if paired['documents'] < 2:
        return 'insufficient_documents'
    if baseline_score is not None and baseline_score <= 0:
        return 'uninformative_baseline'
    lower, upper = paired['ci95']
    if lower >= -allowed_drop:
        return 'meets_configured_screen'
    if upper < -allowed_drop:
        return 'below_configured_screen'
    return 'inconclusive'


def cluster_ci(rows, field, seed=20261010, draws=2000):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[row['document_id']].append(row[field])
    means = [mean(v) for _, v in sorted(groups.items())]
    if not means:
        return None
    rng = random.Random(seed)
    samples = [mean(rng.choices(means, k=len(means))) for _ in range(draws)]
    return dict(documents=len(means), macro_mean=mean(means), ci95=interval(samples), draws=draws, seed=seed)


def summarize(records, manifest):
    groups = defaultdict(list)
    for record in records:
        if record['event'] == 'result':
            groups[record['strategy']].append(record)
    runs = []
    for name, rows in groups.items():
        good = [r for r in rows if r['status'] != 'error']
        usages = [r['usage'] for r in rows if r.get('usage') is not None]
        offline = manifest['config']['backend'] == 'offline'
        expected = len(manifest['question_ids']) * manifest['config'].get('repeats', 1)
        runs.append(dict(strategy=name, requests=len(rows), expected_requests=expected,
                         errors=len(rows)-len(good),
                         empty=sum(r['status'] == 'empty' for r in rows),
                         truncated=sum(r.get('finish_reason') in {'length', 'max_tokens'} for r in rows),
                         cache_hits=sum(r.get('cache_hit', False) for r in rows),
                         unknown_usage_requests=0 if offline else sum(r.get('usage') is None for r in rows),
                         prompt_tokens=None if offline else sum(u['prompt_tokens'] for u in usages),
                         completion_tokens=None if offline else sum(u['completion_tokens'] for u in usages),
                         input_units=sum(r['input_units'] for r in good),
                         baseline_input_units=sum(r['baseline_input_units'] for r in good),
                         input_unit_reduction=1-sum(r['input_units'] for r in good)/max(1,sum(r['baseline_input_units'] for r in good)),
                         actual_context_ratio=mean([r['selected_context_units']/max(1,r['original_context_units']) for r in good]),
                         compression_seconds=mean([r['compression_seconds'] for r in good]),
                         latency=latency_stats(rows),
                         latency_cache_hit=latency_stats([r for r in rows if r.get('cache_hit')]),
                         latency_cache_miss=latency_stats([r for r in rows if not r.get('cache_hit')]),
                         api_latency=latency_stats(rows, 'api_latency_seconds'),
                         usable_answer_rate=sum(r['status'] == 'ok' for r in rows) / len(rows),
                         exact_match=mean([r.get('exact_match') for r in good]),
                         f1=mean([r.get('f1') for r in good]),
                         answer_contains_gold=mean([r.get('answer_contains_gold') for r in good]),
                         answer_span_retained=mean([r.get('answer_span_retained') for r in good]),
                         evidence_retained=mean([r.get('evidence_retained') for r in good]),
                         quality_by_document=cluster_ci(good, 'answer_contains_gold')))
    baseline_name = next((s['name'] for s in manifest['config']['strategies']
                          if s['method'] == 'full' and s.get('cache','none') == 'none'), None)
    baseline = {(r.get('repeat',0),r['id']):r for r in groups.get(baseline_name,[]) if r['status'] != 'error'}
    metric = manifest['config'].get('quality_metric', 'exact_match')
    allowed_drop = manifest['config'].get('allowed_quality_drop', 0)
    for run in runs:
        paired = []
        primary_pairs = []
        for row in groups[run['strategy']]:
            base = baseline.get((row.get('repeat',0),row['id']))
            if base and row['status'] != 'error':
                paired.append(dict(document_id=row['document_id'], delta=float(row['answer_contains_gold'])-float(base['answer_contains_gold'])))
                if row.get(metric) is not None and base.get(metric) is not None:
                    primary_pairs.append(dict(document_id=row['document_id'], delta=float(row[metric])-float(base[metric])))
        run['paired_quality_delta_vs_full'] = cluster_ci(paired, 'delta')
        run['paired_primary_quality_delta_vs_full'] = cluster_ci(primary_pairs, 'delta')
        run['quality_screen'] = quality_screen(manifest['config']['backend'], metric,
                                               run['paired_primary_quality_delta_vs_full'],
                                               run['expected_requests'], len(primary_pairs), run['errors'], allowed_drop,
                                               baseline_score=mean([r.get(metric) for r in baseline.values()]))
    return dict(manifest=manifest, api_attempts=sum(r['event']=='start' for r in records), runs=runs)


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def seconds(value):
    if value is None:
        return '—'
    return f'{value:.6f}' if 0 < value < .0001 else f'{value:.4f}'


def write_report(summary, output):
    output = Path(output)
    config = summary['manifest']['config']
    lines = ['# 配置化实验报告', '',
             f"运行状态：`{summary['status']}`；后端：`{config['backend']}`；数据：`{config['data']}`；划分：`{config.get('split', '无')}`。", '',
             f"数据 SHA256：`{summary['manifest']['data_sha256']}`。", '',
             '上下文上限按配置计数器计算；字符数不是 DeepSeek token。离线句子回答器的质量仅用于诊断检索，不代表大模型表现。', '',
             '评估顺序：先检查任务质量，再比较完整输入/输出消耗与端到端速度。离线诊断结果不能通过真实任务质量门槛。', '',
             '| 策略 | 请求 | EM | F1 | 答案片段包含 | 答案原文保留 | 输入单位减少 | 实际上下文比例 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    order = {s['name']: i for i,s in enumerate(config['strategies'])}
    for r in sorted(summary['runs'], key=lambda r: order[r['strategy']]):
        lines.append(f"| {r['strategy']} | {r['requests']} | {pct(r['exact_match'])} | {pct(r['f1'])} | {pct(r['answer_contains_gold'])} | {pct(r['answer_span_retained'])} | {pct(r['input_unit_reduction'])} | {pct(r['actual_context_ratio'])} |")
    lines += ['', '## 速度与可用性', '',
              '| 策略 | 端到端 P50 秒 | 端到端 P95 秒 | 压缩均值秒 | 命中 P50 秒 | 未命中 P50 秒 | 完整非空输出比例 | 质量筛查状态 |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(summary['runs'], key=lambda r: order[r['strategy']]):
        lines.append(f"| {r['strategy']} | {seconds(r['latency']['p50_seconds'])} | {seconds(r['latency']['p95_seconds'])} | {seconds(r['compression_seconds'])} | {seconds(r['latency_cache_hit']['p50_seconds'])} | {seconds(r['latency_cache_miss']['p50_seconds'])} | {pct(r['usable_answer_rate'])} | {r['quality_screen']} |")
    if config['backend'] == 'deepseek':
        lines += ['', '## 已计量的 API token', '',
                  '| 策略 | 输入 token | 输出 token | 合计 | usage 未知请求 |',
                  '|---|---:|---:|---:|---:|']
        for r in sorted(summary['runs'], key=lambda r: order[r['strategy']]):
            lines.append(f"| {r['strategy']} | {r['prompt_tokens']} | {r['completion_tokens']} | {r['prompt_tokens']+r['completion_tokens']} | {r['unknown_usage_requests']} |")
        lines += ['', '合计只覆盖已取得 usage 的请求；存在未知 usage 时不代表完整账单。服务端缓存输入与输出需按实验日期单价分别计费。']
    lines += ['', '## 统计解释', '',
              '- EM/F1 使用英文 SQuAD 归一化并在多个可接受参考答案中取最大值；合成中文片段数据不报告该指标。',
              '- 答案原文保留只说明答案字符串仍在上下文中，不证明完整证据保留或模型能正确作答。',
              '- summary.json 中记录材料宏平均与材料簇 bootstrap 区间；同篇文章的题目和重复运行不视为独立材料。',
              '- 公共数据按文章隔离，当前比例和邻域半径属于固定对照设置，不根据测试结果选择最佳参数。',
              '- 不同方法按相同名义字符预算运行，实际输入长度仍可能不同；尚未完成相同实际模型 token 预算的比较。',
              '- 空答、截断仍参与质量评分；API 错误单独计数，未知 usage 不视为零费用。失败或不完整运行不能作为完整比较。', '']
    lines += ['- 默认主质量指标为短答案 EM，允许下降为 0；真实生成实验需完整题目配对，且材料簇质量差区间下界满足预设门槛，才标为 meets_configured_screen。',
              '- 该筛查只针对所用自动指标及当前材料；还需人工复核，区间覆盖 0 不等于质量等价。proxy_only 表示仅有离线或片段代理指标。',
              '- 新日志端到端计时从缓存查找到答案准备完成，包含筛选和生成，排除评测与日志写入；旧日志按原计时范围分析，不能和新日志直接比较速度。',
              '- P95 使用样本分位数，少量请求时波动较大。离线速度不代表网络 API 速度；当前非流式接口未测首 token 延迟或并发吞吐量。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = {s['name']:i for i,s in enumerate(summary['manifest']['config']['strategies'])}
    runs = sorted(summary['runs'],key=lambda r:order[r['strategy']])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for prefix,color in [('head-','#d79524'),('bm25-','#236ab9'),('neighbor1-','#168573')]:
        points=sorted([r for r in runs if r['strategy'].startswith(prefix)],key=lambda r:r['input_unit_reduction'])
        axes[0].plot([100*r['input_unit_reduction'] for r in points],[100*r['answer_contains_gold'] for r in points],marker='o',label=prefix[:-1],color=color)
    full=next(r for r in runs if r['strategy']=='full')
    axes[0].scatter([0],[100*full['answer_contains_gold']],color='#333333',marker='D',label='full')
    axes[0].set(xlabel='Input character reduction (%)', ylabel='Answer-fragment containment (%)', title='Offline retrieval diagnostic',xlim=(-4,82),ylim=(0,100))
    axes[0].legend(loc='lower left')
    axes[0].grid(alpha=.2)
    names = [r['strategy'] for r in runs]
    axes[1].barh(names, [100*(r['answer_span_retained'] or 0) for r in runs], color='#167c80')
    axes[1].set(xlabel='Reference answer span retained (%)', xlim=(0,105), title='Context retention (not QA accuracy)')
    fig.savefig(output, dpi=160)
    plt.close(fig)


def historical(csv_path, output):
    """Recompute the 2026-10-09 unique-question analysis from the public CSV."""
    groups = defaultdict(list)
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as handle:
        for r in csv.DictReader(handle):
            r['answer_contains_gold'] = r['gold_fragment_pass'].lower() in {'true','1'}
            groups[r['method']+'+'+r['cache']].append(r)
    result = {}
    doc_means = {}
    for name, rows in groups.items():
        by_doc = defaultdict(list)
        for r in rows:
            by_doc[r['document_id']].append(r['answer_contains_gold'])
        doc_means[name] = {d:mean(v) for d,v in by_doc.items()}
        result[name] = dict(questions=len(rows), prompt_tokens=sum(int(r['prompt_tokens']) for r in rows),
                            completion_tokens=sum(int(r['completion_tokens']) for r in rows),
                            fragment_pass=mean([r['answer_contains_gold'] for r in rows]),
                            **cluster_ci(rows, 'answer_contains_gold', seed=20261009, draws=20000))
    for name in result:
        delta_rows = [dict(document_id=d, delta=v-doc_means['full+none'][d]) for d,v in doc_means[name].items()]
        result[name]['paired_delta'] = cluster_ci(delta_rows, 'delta', seed=20261009, draws=20000)
    output.mkdir(parents=True, exist_ok=True)
    (output/'historical_recomputed.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4), constrained_layout=True)
    names=list(result)
    axes[0].bar(names,[result[n]['prompt_tokens'] for n in names],label='Input',color='#167c80')
    axes[0].bar(names,[result[n]['completion_tokens'] for n in names],bottom=[result[n]['prompt_tokens'] for n in names],label='Output',color='#e6a23c')
    axes[0].set(ylabel='Total provider tokens (56 unique questions)',title='Input and output must both be counted')
    axes[0].tick_params(axis='x',rotation=22)
    axes[0].legend()
    centers=[result[n]['macro_mean']*100 for n in names]
    axes[1].errorbar(range(len(names)),centers,yerr=[[centers[i]-result[n]['ci95'][0]*100 for i,n in enumerate(names)], [result[n]['ci95'][1]*100-centers[i] for i,n in enumerate(names)]],fmt='o',capsize=4,color='#167c80')
    axes[1].set(xticks=range(len(names)),xticklabels=names,ylabel='Document-macro fragment pass (%)',ylim=(0,105),title='Document cluster bootstrap: 95% interval')
    axes[1].tick_params(axis='x',rotation=22)
    fig.savefig(output/'deepseek_tradeoff.png',dpi=160)
    plt.close(fig)
    return result


def historical_latency(log_path, output):
    """Publish timing and numeric fields only; never copy generated text."""
    groups = defaultdict(list)
    records = [json.loads(s) for s in Path(log_path).read_text(encoding='utf-8').splitlines()]
    output.mkdir(parents=True, exist_ok=True)
    fields = ['id', 'document_id', 'method', 'cache', 'cache_hit', 'end_to_end_seconds',
              'api_latency_seconds', 'finish_reason']
    with (output/'deepseek_latency_per_request.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            groups[record['method']+'+'+record['cache']].append(record)
            writer.writerow(record)
    result = {name:dict(requests=len(rows), end_to_end=latency_stats(rows),
                        cache_hit=latency_stats([r for r in rows if r['cache_hit']]),
                        cache_miss=latency_stats([r for r in rows if not r['cache_hit']]),
                        api_non_cached=latency_stats([r for r in rows if not r['cache_hit']], 'api_latency_seconds'))
              for name,rows in groups.items()}
    (output/'deepseek_latency.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# 历史 DeepSeek 速度复核', '',
             '从 2026-10-09 的本地原始日志提取数值；包含全部 74 条请求及应用层缓存命中。', '',
             '| 策略 | 请求 | 端到端 P50 秒 | 端到端 P95 秒 | 命中 P50 秒 | 未命中 P50 秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for name,r in result.items():
        lines.append(f"| {name} | {r['requests']} | {seconds(r['end_to_end']['p50_seconds'])} | {seconds(r['end_to_end']['p95_seconds'])} | {seconds(r['cache_hit']['p50_seconds'])} | {seconds(r['cache_miss']['p50_seconds'])} |")
    lines += ['', '本轮按固定策略顺序各运行一次，未随机交错或重复测量；输入很短，速度受到网络与服务端缓存影响。P95 是当前请求流的样本分位数。',
              '原计时围绕 Pipeline.run，包含筛选和调用，但未独立记录压缩、查缓存及首 token 延迟；与新版 pipeline-v2 的计时范围不同。',
              '旧 CSV 未公开这些耗时字段，现新增数值文件用于复核。不能据单轮耗时证明某方法稳定更快；准确率筛查仍先于效率结论。', '']
    (output/'DEEPSEEK_LATENCY.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--historical-csv',type=Path)
    parser.add_argument('--latency-log',type=Path,help='Optional local historical log; only numerical fields are exported')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.historical_csv:
        historical(args.historical_csv,args.output)
    elif args.run:
        manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
        records=[json.loads(s) for s in (args.run/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        result=summarize(records,manifest)
        previous=json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
        result['status']=previous['status']
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        write_report(result,args.output)
        if args.plot:
            plot_summary(result,args.output/'tradeoff.png')
    else:
        parser.error('--run or --historical-csv is required')
    if args.latency_log:
        historical_latency(args.latency_log, args.output)


if __name__=='__main__':
    main()
