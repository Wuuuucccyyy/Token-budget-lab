"""Synthetic benchmark: gold evidence is used only AFTER inference, for scoring."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .core import CounterBackend, Pipeline


ROOT = Path(__file__).resolve().parents[1]


def load_data(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate request ID")
        seen.add(row["id"])
        if not row["evidence"] or any(e not in row["context"] for e in row["evidence"]):
            raise ValueError(f"invalid evidence: {row['id']}")
        if not row["expected"]:
            raise ValueError("expected answer fragments required")
    if not rows:
        raise ValueError("empty dataset")
    return rows


def evaluate(rows: list[dict], counter: CounterBackend, method: str, cache: str,
             ratio: float, threshold: float) -> dict:
    pipe = Pipeline(counter, method=method, cache=cache, ratio=ratio, threshold=threshold)
    details = []
    labels = {row["id"]: (row["namespace"], row["context"], row["fact_id"]) for row in rows}
    for row in rows:
        # Only query, context, ID and namespace go into the pipeline. No gold labels.
        result = pipe.run(row["query"], row["context"], row["id"], row["namespace"])
        result["id"] = row["id"]
        result["query"] = row["query"]
        result["evidence_retained"] = all(e in result["selected_context"] for e in row["evidence"])
        result["answer_contains_gold"] = all(s in result["answer"] for s in row["expected"])
        result["wrong_reuse"] = bool(result["hit"] and labels[result["cache_source"]] != labels[row["id"]])
        details.append(result)
    total = len(details)
    baseline = sum(d["baseline_units"] for d in details)
    sent = sum(d["sent_units"] for d in details)
    hits = sum(d["hit"] for d in details)
    wrong = sum(d["wrong_reuse"] for d in details)
    return dict(method=method, cache=cache, ratio=ratio, threshold=threshold,
                requests=total, simulated_calls=total - hits, hits=hits, wrong_reuse=wrong,
                cache_error_rate=wrong / hits if hits else None,
                baseline_input_units=baseline, simulated_input_units=sent,
                simulated_input_reduction=1 - sent / baseline if baseline else 0.0,
                evidence_retention=sum(d["evidence_retained"] for d in details) / total,
                offline_answer_contains_gold=sum(d["answer_contains_gold"] for d in details) / total,
                details=details)


def write_report(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 离线实验报告", "", "本报告由命令行运行生成。数据为人工构造的合成样例，回答器是句子检索器，不是大模型。",
             f"计数单位：**{result['unit']}**；编码：`{result['encoding']}`；Python：{result['python']}。",
             "输入统计包含本项目提示词模板，但不包含服务商消息封装；没有真实 API 调用、计费或延迟测量。",
             "", "| 策略 | 保留比例 | 阈值 | 模拟调用数 | 模拟输入减少 | 全部证据保留率 | 离线答案包含率 | 错误复用/命中 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in result["runs"]:
        lines.append(f"| {run['method']} + {run['cache']} | {run['ratio']:.0%} | {run['threshold']:.2f} | "
                     f"{run['simulated_calls']}/{run['requests']} | {run['simulated_input_reduction']:.1%} | "
                     f"{run['evidence_retention']:.1%} | {run['offline_answer_contains_gold']:.1%} | "
                     f"{run['wrong_reuse']}/{run['hits']} |")
    lines += ["", "## 怎么读", "",
              "- 模拟输入减少 = 1 − 各请求实际拟发送输入单位之和 / 所有请求完整输入单位之和。缓存命中拟发送量为 0。",
              "- 全部证据保留率：某条问题标注的所有证据原文是否都保留；缓存命中时检查最初生成缓存答案时的上下文。",
              "- 离线答案包含率：检索器返回的句子是否包含所有标准答案片段。不是 LLM 准确率，也不能衡量推理或否定是否正确。",
              "- 错误复用：命中了不同 fact_id 的问题；这个标签只供评测使用。相同 fact_id 的错误答案仍可能被复用。",
              "- 固定顺序中包含刻意重复和改写的问题，命中率受它们的比例及顺序影响，不能外推真实流量。",
              "- head 与 bm25 都以相同比例的原始上下文长度为预算，保留整句，因此实际长度可能小于预算。",
              "- 不同预算下的缓存分别重置；本实验没有调参后再声称独立测试集结果。",
              "", "## 下一步", "",
              "增加独立数据、分离验证/测试集；接入真实回答模型并记录 usage、输出 token 和端到端延迟；加入 LLMLingua-2 基线。",
              "对 lexical 阈值做扫描，检查数字、否定和近似问题是否导致错误复用。",
              "本结果不能证明超越 GPTCache 或 LLMLingua，二者尚未在此运行。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="零依赖离线压缩与缓存实验；不调用大模型。")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "toy_qa.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--encoding", default="characters", help="characters 或 tiktoken 编码名称")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.9, 0.98])
    args = parser.parse_args()
    if any(not 0 < r <= 1 for r in args.ratios) or any(not 0 <= t <= 1 for t in args.thresholds):
        parser.error("ratios must be in (0, 1]; thresholds must be in [0, 1]")
    counter = CounterBackend(args.encoding)
    rows = load_data(args.data)
    configs = [("full", "none", 1.0, 0.9), ("full", "exact", 1.0, 0.9)]
    for ratio in args.ratios:
        configs.extend((method, cache, ratio, 0.9) for method, cache in
                       [("head", "none"), ("bm25", "none"), ("bm25", "exact")])
        configs.extend(("bm25", "lexical", ratio, t) for t in args.thresholds)
    result = dict(unit=counter.unit, encoding=counter.name, python=platform.python_version(),
                  dataset="synthetic QA workload", backend="offline-extractive-v1",
                  runs=[evaluate(rows, counter, *config) for config in configs])
    write_report(result, args.output)
    print(f"{len(rows)} requests; {len(configs)} configurations; unit={counter.unit}")
    print(f"Report: {(args.output / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
