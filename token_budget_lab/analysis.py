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


def interval(values):
    values = sorted(values)
    def quantile(p):
        index = (len(values)-1)*p
        lower = int(index)
        return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)
    return [quantile(.025), quantile(.975)]


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
        runs.append(dict(strategy=name, requests=len(rows), errors=len(rows)-len(good),
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
                         exact_match=mean([r.get('exact_match') for r in good]),
                         f1=mean([r.get('f1') for r in good]),
                         answer_contains_gold=mean([r.get('answer_contains_gold') for r in good]),
                         answer_span_retained=mean([r.get('answer_span_retained') for r in good]),
                         evidence_retained=mean([r.get('evidence_retained') for r in good]),
                         quality_by_document=cluster_ci(good, 'answer_contains_gold')))
    baseline_name = next((s['name'] for s in manifest['config']['strategies']
                          if s['method'] == 'full' and s.get('cache','none') == 'none'), None)
    baseline = {(r.get('repeat',0),r['id']):r for r in groups.get(baseline_name,[]) if r['status'] != 'error'}
    for run in runs:
        paired = []
        for row in groups[run['strategy']]:
            base = baseline.get((row.get('repeat',0),row['id']))
            if base and row['status'] != 'error':
                paired.append(dict(document_id=row['document_id'], delta=float(row['answer_contains_gold'])-float(base['answer_contains_gold'])))
        run['paired_quality_delta_vs_full'] = cluster_ci(paired, 'delta')
    return dict(manifest=manifest, api_attempts=sum(r['event']=='start' for r in records), runs=runs)


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def write_report(summary, output):
    output = Path(output)
    config = summary['manifest']['config']
    lines = ['# 配置化实验报告', '',
             f"运行状态：`{summary['status']}`；后端：`{config['backend']}`；数据：`{config['data']}`；划分：`{config.get('split', '无')}`。", '',
             f"数据 SHA256：`{summary['manifest']['data_sha256']}`。", '',
             '上下文上限按配置计数器计算；字符数不是 DeepSeek token。离线句子回答器的质量仅用于诊断检索，不代表大模型表现。', '',
             '| 策略 | 请求 | 实际上下文比例 | 输入单位减少 | 答案片段包含 | EM | F1 | 答案原文保留 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    order = {s['name']: i for i,s in enumerate(config['strategies'])}
    for r in sorted(summary['runs'], key=lambda r: order[r['strategy']]):
        lines.append(f"| {r['strategy']} | {r['requests']} | {pct(r['actual_context_ratio'])} | {pct(r['input_unit_reduction'])} | {pct(r['answer_contains_gold'])} | {pct(r['exact_match'])} | {pct(r['f1'])} | {pct(r['answer_span_retained'])} |")
    lines += ['', '## 统计解释', '',
              '- EM/F1 使用英文 SQuAD 归一化并在多个可接受参考答案中取最大值；合成中文片段数据不报告该指标。',
              '- 答案原文保留只说明答案字符串仍在上下文中，不证明完整证据保留或模型能正确作答。',
              '- summary.json 中记录材料宏平均与材料簇 bootstrap 区间；同篇文章的题目和重复运行不视为独立材料。',
              '- 公共数据按文章隔离，当前比例和邻域半径属于固定对照设置，不根据测试结果选择最佳参数。',
              '- 不同方法按相同名义字符预算运行，实际输入长度仍可能不同；尚未完成相同实际模型 token 预算的比较。',
              '- 空答、截断仍参与质量评分；API 错误单独计数，未知 usage 不视为零费用。失败或不完整运行不能作为完整比较。', '']
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--historical-csv',type=Path)
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


if __name__=='__main__':
    main()
