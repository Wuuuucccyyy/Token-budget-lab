"""Read this file first. Standard library only unless a tokenizer is requested."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable


class CounterBackend:
    def __init__(self, encoding: str = "characters"):
        self.name = encoding
        self.encoder = None
        if encoding != "characters":
            import tiktoken  # Optional; missing dependency is an error, never a silent fallback.
            self.encoder = tiktoken.get_encoding(encoding)

    @property
    def unit(self) -> str:
        return "characters" if self.encoder is None else "tokens"

    def count(self, text: str) -> int:
        if self.encoder is None:
            return len(text)
        return len(self.encoder.encode(text, disallowed_special=()))


def terms(text: str) -> list[str]:
    """English words plus Chinese characters/bigrams; not an LLM tokenizer."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run)
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words


def sentences(text: str) -> list[str]:
    # Educational splitter; abbreviations and tables need a better parser.
    return [x.strip() for x in re.findall(r"[^。！？.!?\n]+(?:[。！？.!?]+|$)", text, re.M) if x.strip()]


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    docs = [Counter(terms(text)) for text in texts]
    avg = sum(sum(doc.values()) for doc in docs) / max(len(docs), 1)
    df = Counter(term for doc in docs for term in doc)
    scores = []
    for doc in docs:
        score = 0.0
        for term in set(terms(query)):
            tf = doc[term]
            if tf:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * sum(doc.values()) / (avg or 1)))
        scores.append(score)
    return scores


def compress(query: str, context: str, budget: int, counter: CounterBackend,
             method: str = "bm25") -> str:
    """Keep whole sentences, within a context-only budget, in original order."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if method not in {"head", "bm25"}:
        raise ValueError("unknown compression method")
    if counter.count(context) <= budget:
        return context
    parts = sentences(context)
    scores = bm25_scores(query, parts)
    order = list(range(len(parts)))
    if method == "bm25":
        order.sort(key=lambda i: (-scores[i], i))
    selected: list[int] = []
    for index in order:
        candidate = sorted(selected + [index])
        if counter.count("\n".join(parts[i] for i in candidate)) <= budget:
            selected.append(index)
        elif method == "head":
            break
    return "\n".join(parts[i] for i in sorted(selected))


SYSTEM = "仅根据资料回答问题；资料不足时回答：证据不足。"


def prompt(query: str, context: str, system: str = SYSTEM) -> str:
    return f"{system}\n资料：\n{context}\n问题：{query}\n回答："


def extractive_answer(query: str, context: str) -> str:
    """Offline surrogate: return the highest-scoring sentence. NOT a language model."""
    parts = sentences(context)
    if not parts:
        return "证据不足"
    scores = bm25_scores(query, parts)
    index = max(range(len(parts)), key=lambda i: scores[i])
    return parts[index] if scores[index] > 0 else "证据不足"


def fingerprint(context: str, namespace: str, model: str, system: str, config: str) -> str:
    # Identical questions in different documents/users/configurations must not collide.
    value = json.dumps([context, namespace, model, system, config], ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def similarity(a: str, b: str) -> float:
    """Lexical cosine similarity. This is NOT embedding-based semantic similarity."""
    x, y = Counter(terms(a)), Counter(terms(b))
    denominator = math.sqrt(sum(v * v for v in x.values()) * sum(v * v for v in y.values()))
    return sum(v * y[k] for k, v in x.items()) / denominator if denominator else 0.0


@dataclass
class CacheEntry:
    query: str
    answer: str
    source_context: str
    request_id: str


class AnswerCache:
    """In-memory, single-process cache. Exact mode is the default."""
    def __init__(self, mode: str = "exact", threshold: float = 0.9):
        if mode not in {"none", "exact", "lexical"}:
            raise ValueError("invalid cache mode")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.mode, self.threshold = mode, threshold
        self.entries: dict[str, list[CacheEntry]] = {}

    def lookup(self, scope: str, query: str) -> CacheEntry | None:
        if self.mode == "none":
            return None
        candidates = self.entries.get(scope, [])
        for entry in candidates:
            if entry.query == query:
                return entry
        if self.mode == "lexical" and candidates:
            best = max(candidates, key=lambda e: similarity(query, e.query))
            if similarity(query, best.query) >= self.threshold:
                return best
        return None

    def put(self, scope: str, entry: CacheEntry) -> None:
        if self.mode != "none":
            self.entries.setdefault(scope, []).append(entry)


class Pipeline:
    def __init__(self, counter: CounterBackend, method: str = "bm25", ratio: float = 0.5,
                 cache: str = "exact", threshold: float = 0.9,
                 answer_fn: Callable[[str, str], str] = extractive_answer,
                 model: str = "offline-extractive-v1"):
        if method not in {"full", "head", "bm25"}:
            raise ValueError("invalid method")
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.counter, self.method, self.ratio = counter, method, ratio
        self.cache = AnswerCache(cache, threshold)
        self.answer_fn, self.model = answer_fn, model

    def run(self, query: str, context: str, request_id: str = "", namespace: str = "demo",
            system: str = SYSTEM) -> dict:
        config = json.dumps([self.method, self.ratio, self.counter.name])
        scope = fingerprint(context, namespace, self.model, system, config)
        baseline = self.counter.count(prompt(query, context, system))
        hit = self.cache.lookup(scope, query)
        if hit:
            return dict(answer=hit.answer, selected_context=hit.source_context, hit=True,
                        cache_source=hit.request_id, baseline_units=baseline, sent_units=0,
                        context_units=self.counter.count(hit.source_context))
        selected = context if self.method == "full" else compress(
            query, context, math.floor(self.counter.count(context) * self.ratio),
            self.counter, self.method)
        answer = self.answer_fn(query, selected)
        self.cache.put(scope, CacheEntry(query, answer, selected, request_id))
        return dict(answer=answer, selected_context=selected, hit=False, cache_source=None,
                    baseline_units=baseline, sent_units=self.counter.count(prompt(query, selected, system)),
                    context_units=self.counter.count(selected))
