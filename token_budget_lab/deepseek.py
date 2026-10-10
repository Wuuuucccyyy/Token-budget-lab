"""Optional paid experiment. Uses urllib and DEEPSEEK_API_KEY, no SDK required."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import ROOT, load_data
from .core import CounterBackend, Pipeline, prompt


ENDPOINT = "https://api.deepseek.com/chat/completions"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("API redirect rejected; check the official endpoint")


class DeepSeek:
    def __init__(self, model: str, max_tokens: int = 256, timeout: float = 120,
                 protocol: str = "chat", thinking: str = "default", allow_empty: bool = False,
                 answer_instruction: str = ""):
        self.key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.key:
            raise ValueError("Set DEEPSEEK_API_KEY locally; do not put it in code or chat")
        if not model.strip() or max_tokens < 1 or timeout <= 0:
            raise ValueError("model, max_tokens and timeout must be valid")
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout
        if protocol not in {"chat", "anthropic"} or thinking not in {"default", "disabled"}:
            raise ValueError("invalid protocol or thinking setting")
        self.protocol, self.thinking, self.allow_empty = protocol, thinking, allow_empty
        self.answer_instruction = answer_instruction
        self.opener = urllib.request.build_opener(NoRedirect())
        self.last: dict = {}

    def __call__(self, query: str, context: str) -> str:
        self.last = {}
        content = prompt(query, context)
        if self.answer_instruction:
            content += '\n' + self.answer_instruction
        body = dict(model=self.model, messages=[{"role": "user", "content": content}],
                    stream=False, max_tokens=self.max_tokens)
        endpoint = ENDPOINT
        headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        if self.protocol == "anthropic":
            endpoint = "https://api.deepseek.com/anthropic/v1/messages"
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
        if self.thinking == "disabled":
            if self.protocol == "chat":
                body["thinking"] = {"type": "disabled"}
            else:
                body["reasoning"] = {"effort": "none"}
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not print request headers, keys, or remote error bodies.
            raise RuntimeError(f"DeepSeek HTTP {exc.code}; no automatic retry") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("DeepSeek network error; no automatic retry") from None
        elapsed = time.perf_counter() - start
        usage = payload.get("usage", {})
        if self.protocol == "anthropic":
            usage = dict(usage, prompt_tokens=usage.get("input_tokens"),
                         completion_tokens=usage.get("output_tokens"),
                         prompt_cache_hit_tokens=usage.get("cache_read_input_tokens", 0))
        for field in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage.get(field), int) or usage[field] < 0:
                raise RuntimeError("Response missing valid usage; token totals unavailable")
        # Preserve known billable usage even when the answer schema is malformed.
        self.last = dict(usage=usage, api_latency_seconds=elapsed, response_model=payload.get("model"))
        if self.protocol == "chat":
            choice = payload["choices"][0]
            text = choice["message"].get("content")
            finish = choice.get("finish_reason")
        else:
            text = ''.join(b.get('text', '') for b in payload.get('content', []) if b.get('type') == 'text')
            finish = payload.get('stop_reason')
        self.last = dict(usage=usage, api_latency_seconds=elapsed, response_model=payload.get("model"),
                         finish_reason=finish)
        if not isinstance(text, str) or not text.strip():
            if self.allow_empty:
                return ""
            raise RuntimeError("No answer content; increase max_tokens or check model settings")
        return text


def live_experiment(rows: list[dict], client: DeepSeek, ratio: float, output: Path) -> dict:
    """Each strategy has a fresh cache and processes the same ordered requests."""
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "requests.jsonl"
    if log_path.exists() or (output / "summary.json").exists():
        raise ValueError("Output already contains an experiment; use a new output directory")
    runs = []
    configs = [("full", "none"), ("head", "none"), ("bm25", "none"),
               ("full", "exact"), ("bm25", "exact")]
    summary = dict(status="running", requested_model=client.model, max_tokens=client.max_tokens,
                   context_budget_unit="characters", ratio=ratio, rows=len(rows), runs=runs)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        with log_path.open("x", encoding="utf-8") as log:
            for method, cache in configs:
                pipe = Pipeline(CounterBackend(), method=method, ratio=ratio, cache=cache,
                                answer_fn=client, model=client.model)
                details = []
                for row in rows:
                    client.last = {}
                    start = time.perf_counter()
                    result = pipe.run(row["query"], row["context"], row["id"], row["namespace"])
                    record = dict(id=row["id"], method=method, cache=cache, cache_hit=result["hit"],
                                  cache_source=result["cache_source"], answer=result["answer"],
                                  selected_context=result["selected_context"],
                                  end_to_end_seconds=time.perf_counter() - start,
                                  evidence_retained=all(e in result["selected_context"] for e in row["evidence"]),
                                  answer_contains_gold=all(s in result["answer"] for s in row["expected"]))
                    record.update(client.last if not result["hit"] else {
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                        "api_latency_seconds": 0, "finish_reason": "cache"})
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()  # Keep completed paid calls if a later request fails.
                    details.append(record)
                runs.append(dict(method=method, cache=cache,
                                 api_calls=sum(not d["cache_hit"] for d in details),
                                 prompt_tokens=sum(d["usage"]["prompt_tokens"] for d in details),
                                 completion_tokens=sum(d["usage"]["completion_tokens"] for d in details),
                                 provider_cache_hit_tokens=sum(d["usage"].get("prompt_cache_hit_tokens", 0) for d in details),
                                 total_end_to_end_seconds=sum(d["end_to_end_seconds"] for d in details),
                                 answer_contains_gold=sum(d["answer_contains_gold"] for d in details) / len(details),
                                 truncated_answers=sum(d.get("finish_reason") == "length" for d in details)))
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        summary["status"] = "incomplete"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    baseline = runs[0]["prompt_tokens"]
    for run in runs:
        run["input_token_reduction"] = 1 - run["prompt_tokens"] / baseline if baseline else 0
    summary["status"] = "complete"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paid DeepSeek comparison; at most 5 × limit API calls; no retries.")
    parser.add_argument("--model", required=True, help="A model ID currently available in your DeepSeek account")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "toy_qa.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 1 or not 0 < args.ratio <= 1:
        parser.error("limit must be positive and ratio must be in (0, 1]")
    rows = load_data(args.data)[:args.limit]
    output = args.output or ROOT / "local_results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    client = DeepSeek(args.model, args.max_tokens, args.timeout)
    summary = live_experiment(rows, client, args.ratio, output)
    print(f"Completed {sum(r['api_calls'] for r in summary['runs'])} API calls. Results: {output.resolve()}")


if __name__ == "__main__":
    main()
