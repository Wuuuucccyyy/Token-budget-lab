import unittest

from token_budget_lab.core import (
    AnswerCache, CacheEntry, CounterBackend, Pipeline, compress, fingerprint, sentences,
)
from token_budget_lab.benchmark import evaluate, load_data, ROOT


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.counter = CounterBackend()

    def test_query_relevant_fact_at_end(self):
        context = "花园里有一棵树。会议室有投影仪。图书馆周日闭馆。"
        output = compress("图书馆周日开放吗", context, 10, self.counter)
        self.assertEqual(output, "图书馆周日闭馆。")

    def test_budget_and_sentence_integrity(self):
        context = "甲实验室周日不开门。乙实验室周日开放。甲实验室在三楼。"
        for budget in range(len(context) + 1):
            output = compress("甲实验室周日", context, budget, self.counter)
            self.assertLessEqual(len(output), budget)
            self.assertTrue(all(s in sentences(context) for s in sentences(output)))

    def test_original_order(self):
        context = "第一条苹果。无关信息很多很多很多。第二条苹果。"
        self.assertEqual(compress("苹果", context, 15, self.counter), "第一条苹果。\n第二条苹果。")

    def test_empty_and_impossible_budget(self):
        self.assertEqual(compress("问题", "", 0, self.counter), "")
        self.assertEqual(compress("问题", "一句话。", 1, self.counter), "")

    def test_whole_context_preserves_format(self):
        text = "甲。\n\n乙。"
        self.assertEqual(compress("甲", text, len(text), self.counter), text)


class CacheTests(unittest.TestCase):
    def test_exact_hit_skips_answer_function(self):
        calls = []
        def answer(q, c):
            calls.append(q)
            return "结果"
        pipe = Pipeline(CounterBackend(), answer_fn=answer)
        first = pipe.run("问题", "资料内容。", "a")
        second = pipe.run("问题", "资料内容。", "b")
        self.assertFalse(first["hit"])
        self.assertTrue(second["hit"])
        self.assertEqual(second["sent_units"], 0)
        self.assertEqual(second["cache_source"], "a")
        self.assertEqual(len(calls), 1)

    def test_no_cross_context_or_user_or_system_hits(self):
        pipe = Pipeline(CounterBackend())
        pipe.run("开放吗", "周日开放。")
        self.assertFalse(pipe.run("开放吗", "周日关闭。")["hit"])
        self.assertFalse(pipe.run("开放吗", "周日开放。", namespace="another-user")["hit"])
        self.assertFalse(pipe.run("开放吗", "周日开放。", system="请用英文回答")["hit"])

    def test_scope_model_and_config(self):
        base = fingerprint("c", "n", "m", "s", "v1")
        self.assertNotEqual(base, fingerprint("c", "n", "m2", "s", "v1"))
        self.assertNotEqual(base, fingerprint("c", "n", "m", "s", "v2"))

    def test_exact_preserves_case_and_punctuation(self):
        cache = AnswerCache()
        cache.put("scope", CacheEntry("A?", "yes", "ctx", "id"))
        self.assertIsNone(cache.lookup("scope", "a?"))
        self.assertIsNone(cache.lookup("scope", "A!"))

    def test_lexical_can_confuse_negation(self):
        cache = AnswerCache("lexical", 0.7)
        cache.put("s", CacheEntry("图书馆周日开放吗", "否", "c", "i"))
        self.assertIsNotNone(cache.lookup("s", "图书馆周日不开放吗"))

    def test_disabled_cache(self):
        pipe = Pipeline(CounterBackend(), cache="none")
        for _ in range(2):
            self.assertFalse(pipe.run("问题", "资料。")["hit"])

    def test_invalid_config(self):
        for value in [0, -1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                Pipeline(CounterBackend(), ratio=value)
        with self.assertRaises(ValueError):
            AnswerCache(threshold=2)


class EvaluationTests(unittest.TestCase):
    def test_synthetic_data_valid_and_reproducible(self):
        rows = load_data(ROOT / "data" / "toy_qa.jsonl")
        a = evaluate(rows, CounterBackend(), "bm25", "exact", 0.5, 0.9)
        b = evaluate(rows, CounterBackend(), "bm25", "exact", 0.5, 0.9)
        self.assertEqual(a, b)
        self.assertGreater(a["hits"], 0)
        self.assertEqual(a["wrong_reuse"], 0)

    def test_full_baseline_has_no_savings(self):
        rows = load_data(ROOT / "data" / "toy_qa.jsonl")
        report = evaluate(rows, CounterBackend(), "full", "none", 1, 0.9)
        self.assertEqual(report["simulated_input_reduction"], 0)
        self.assertEqual(report["evidence_retention"], 1)

    def test_gold_never_used_in_inference(self):
        rows = load_data(ROOT / "data" / "toy_qa.jsonl")
        a = evaluate(rows, CounterBackend(), "bm25", "none", 0.5, 0.9)
        changed = [dict(row, evidence=["不存在的证据"], expected=["错误标签"]) for row in rows]
        b = evaluate(changed, CounterBackend(), "bm25", "none", 0.5, 0.9)
        self.assertEqual([x["answer"] for x in a["details"]], [x["answer"] for x in b["details"]])


if __name__ == "__main__":
    unittest.main()
