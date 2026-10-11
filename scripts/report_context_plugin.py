"""Export synthetic v0.6 results, including failures and warm-probe costs."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from token_budget_lab.cache_cost import cache_cost
from token_budget_lab.tool_experiment import export


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--diagnostic',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    export(args.run,dest/'deepseek_test')
    summary=json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(l) for l in (args.run/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=summary['manifest'];rates=manifest['config']['usd_per_million']
    by={r['mode']:r for r in summary['runs']}
    responses=[r for r in rows if r['event']=='response']
    # Original summary is retained. The first run's inherited counter counted
    # history_* only; derive context_* requests from the paid response journal.
    counters={a:sum(c['function']['name'].startswith(('context_','history_')) for r in responses if r['mode']==a
                   for c in r.get('message',{}).get('tool_calls',[])) for a in by}
    write(dest/'recovery_call_accounting.json',dict(counts=counters,source='all paid response tool_calls',
        correction='original summary counter omitted context_*; aggregation fixed after run; API trajectories unchanged'))
    snapshot=args.run/'execution_sources'
    originals={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text(encoding='utf-8')) for p in snapshot.glob('*.py')}
    write(dest/'source_snapshot.json',dict(code_sha256=manifest['code_sha256'],files=originals,
        note='exact pre-execution package files; after-run changes only corrected recovery request counting'))
    first={}
    for r in responses:first.setdefault((r['episode_id'],r['mode']),r)
    probes=[r for r in rows if r['event']=='cache_probe']
    probe_pairs=[]
    for r in probes:
        original=first[r['episode_id'],r['mode']]
        probe_pairs.append(dict(episode_id=r['episode_id'],mode=r['mode'],first_usage=original.get('usage'),warm_usage=r.get('usage'),
            first_api_seconds=original.get('api_latency_seconds'),warm_seconds=r.get('end_to_end_seconds'),tools_executed=False))
    write(dest/'cache_probes.json',probe_pairs)
    write(dest/'cost_accounting.json',dict(rates=rates,source=manifest['pricing_source'],
        interpretation=manifest['pricing_interpretation'],task_runs=summary['runs'],warm_probe_accounting=summary['cache_probes'],
        total_attempts=summary['total_api_attempts_including_probes'],probe_attempts=summary['probe_api_attempts'],
        unknown_usage_attempts=summary['unknown_usage_attempts_including_probes'],research_usage=summary['research_usage_including_probes']))
    failures=[]
    events=[]
    for r in rows:
        if r['event']=='task_result' and not r['task_success']:
            chain=[x for x in rows if x.get('episode_id')==r['episode_id'] and x.get('mode')==r['mode'] and x.get('checkpoint')==r['checkpoint']]
            failures.append(dict(result=r,model_responses=[dict(step=x['step'],message=x.get('message'),status=x['status'],history_mode=x['history_mode']) for x in chain if x['event']=='response'],
                business_actions=[x for x in chain if x['event']=='business_tool'],controller_events=[x for x in chain if x['event']=='recovery_guard']))
        if r['event']=='recovery_guard' and r.get('memory_tool_events'):events.append(r)
    write(dest/'failure_cases.json',failures);write(dest/'controller_events.json',events)
    comparisons=[]
    for a,b in [('context','full'),('context_guarded','full_guarded'),('context_guarded','full')]:
        x,y=by[a],by[b]
        total=lambda r:r['prompt_tokens']+r['completion_tokens']
        comparisons.append(dict(plugin=a,baseline=b,total_token_reduction=1-total(x)/total(y),
            estimated_cost_reduction=None if x['estimated_usd'] is None or not y['estimated_usd'] else 1-x['estimated_usd']/y['estimated_usd'],
            quality_delta=x['task_success_rate']-y['task_success_rate']))
    write(dest/'comparisons.json',comparisons)
    diagnostic=None
    if args.diagnostic:
        d=json.loads((args.diagnostic/'summary.json').read_text(encoding='utf-8'))
        export(args.diagnostic,dest/'network_diagnostic')
        diagnostic=dict(attempts=d['total_api_attempts_including_probes'],unknown_usage=d['unknown_usage_attempts_including_probes'],
            interpretation='network-restricted launch; no response usage; not silently counted as free')
    write(dest/'usage_ledger.json',dict(primary_attempts=summary['total_api_attempts_including_probes'],
        primary_unknown_usage=summary['unknown_usage_attempts_including_probes'],network_diagnostic=diagnostic,
        total_attempts=summary['total_api_attempts_including_probes']+(diagnostic['attempts'] if diagnostic else 0),
        known_usage_repricing=summary['research_usage_including_probes'],
        total_research_invoice=None,invoice_interpretation='not queried; preceding unknown usage retained'))
    def table(metric,formatting):return '| '+metric+' | '+' | '.join(formatting(by[a]) for a in by)+' |'
    lines=['# 工具结果外置、版本核验与缓存费用实验','',
        'v0.6，2026-10-12。八个新合成实例、四种配置、每实例三个连续检查点。更新、时序、拒答沿用已有模板，工具执行新增当前大日志字段恢复；不是独立自然任务基准。',
        '', '## 固定实现结果','',
        '| 指标 | 原生 Full | Full + 来源核验 | 上下文插件 | 插件 + 来源核验 |',
        '|---|---:|---:|---:|---:|',
        table('检查点通过',lambda r:f"{round(r['task_success_rate']*r['checkpoints'])}/{r['checkpoints']}"),
        table('完整任务通过',lambda r:f"{round(r['episode_success_rate']*8)}/8"),
        table('答案准确率',lambda r:f"{r['answer_accuracy']:.1%}"),
        table('文件状态准确率',lambda r:f"{r['artifact_state_accuracy']:.1%}"),
        table('模型调用',lambda r:str(r['model_calls'])),
        table('输入输出总 token',lambda r:f"{r['prompt_tokens']+r['completion_tokens']:,}"),
        table('缓存命中输入比例',lambda r:f"{r['cache_hit_fraction']:.1%}" if r['cache_hit_fraction'] is not None else '未知'),
        table('缓存未命中输入 token',lambda r:f"{r['cache_miss_tokens']:,}"),
        table('固定非高峰价重估 USD',lambda r:f"${r['estimated_usd']:.6f}" if r['estimated_usd'] is not None else '未知'),
        table('检查点 P95 秒',lambda r:f"{r['latency']['p95_seconds']:.3f}"),
        '', '所有恢复、回退和失败调用均计入任务成本；额外缓存重放不增加任务质量样本。估算费用与到账账单不同。',
        '', '原始 summary.json 的恢复计数继承旧接口，只统计 history_*，遗漏 context_*。正确计数由真实响应重算，见 [恢复调用核算](recovery_call_accounting.json)。公开 [执行源码快照](source_snapshot.json) 保留调用前的精确文件；运行后仅修正该汇总计数，没有重跑或改变插件行为。',
        '', '## 同能力对照','']
    for c in comparisons:
        cost='未知' if c['estimated_cost_reduction'] is None else f"{c['estimated_cost_reduction']:.1%}"
        lines.append(f"- {c['plugin']} 相对 {c['baseline']}：总 token 减少 {c['total_token_reduction']:.1%}，固定价格估算费用减少 {cost}，检查点质量差 {c['quality_delta']:+.1%}。")
    lines+=['','Full + 来源核验和插件 + 来源核验使用相同写入合同，单独检验外置/筛选的效果。核验读取真实快照及当前问题，未使用 gold；尚不覆盖任意自然语言推理和写工具。',
        '', '## 前缀与缓存控制','',
        '同一任务内，旧视图与工具定义保持稳定，只有新消息追加。每个实例/组的首部实验标识独立，减少此前请求和跨组缓存预热。没有清空服务端缓存；首请求仅称隔离前缀请求。新 user 任务可能重新选历史，仍需实际 usage 判断收益。',
        '',f"完整任务之外执行 {summary['probe_api_attempts']} 次原样首请求重放，不执行业务工具。原请求与重放的缓存分项见 [cache_probes.json](cache_probes.json)。主轮累计 {summary['total_api_attempts_including_probes']} 次尝试，未知 usage {summary['unknown_usage_attempts_including_probes']} 次。",'']
    if diagnostic:lines.append(f"另保留联网受限的启动诊断：{diagnostic['attempts']} 次尝试、{diagnostic['unknown_usage']} 次未知 usage。累计研究尝试 {summary['total_api_attempts_including_probes']+diagnostic['attempts']} 次；未知项不填零成本。")
    lines+=['',f"主轮包含重放的已知 usage 按固定价格重估约 ${summary['research_usage_including_probes']['estimated_usd']:.6f}，不是账户扣费。价格为核对日 [DeepSeek 官方 Flash 非高峰档](https://api-docs.deepseek.com/quick_start/pricing/)；高峰档两倍。",'',
        '## 失败与验证边界','',f"共保留 {len(failures)} 个失败检查点。失败原因、实际操作和后续状态见 [失败记录](failure_cases.json)；拒绝写入和检索事件见 [控制事件](controller_events.json)。",'',
        '每实例仍最多六次模型调用。没有自动重试、调大步骤上限、删除失败或修复中间状态。本轮是固定实现的一次专项试验；八个实例来自四类模板，样本小，不能证明一般场景近乎无损。核验收益不能全部归因于压缩。',
        '', '真正发生压缩的任务及匹配 Full 对照另见 summary.json 的 compressed_subset；该子集是事后诊断，失败与全部调用仍保留。',
        '', '完整数值：[逐检查点](deepseek_test/per_checkpoint.csv) · [逐任务](deepseek_test/per_episode.csv) · [逐请求](deepseek_test/per_request.csv) · [费用核算](cost_accounting.json) · [总调用账本](usage_ledger.json)。',
        '', '复现协议与接入：[CONTEXT_PLUGIN.md](../../docs/CONTEXT_PLUGIN.md)。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(dest),failures=len(failures),comparisons=comparisons),ensure_ascii=True))


if __name__=='__main__':main()
