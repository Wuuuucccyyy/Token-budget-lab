import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from token_budget_lab.deepseek import DeepSeek, live_experiment
from token_budget_lab.benchmark import load_data, ROOT


class DeepSeekTests(unittest.TestCase):
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-placeholder"})
    def test_usage_and_payload_via_mock_transport(self):
        client = DeepSeek("test-model")
        payload = {"model": "test-model", "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 23, "completion_tokens": 4, "prompt_cache_hit_tokens": 10}}
        with patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(payload).encode())) as opened:
            self.assertEqual(client("问题", "资料"), "答案")
        sent = json.loads(opened.call_args.args[0].data)
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["max_tokens"], 256)
        self.assertIn("问题", sent["messages"][0]["content"])
        self.assertEqual(client.last["usage"]["prompt_tokens"], 23)

    @patch.dict(os.environ, {}, clear=True)
    def test_key_required(self):
        with self.assertRaises(ValueError):
            DeepSeek("test-model")

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-placeholder"})
    def test_missing_usage_fails(self):
        client = DeepSeek("test-model")
        with patch.object(client.opener, "open", return_value=io.BytesIO(b'{}')):
            with self.assertRaises(RuntimeError):
                client("q", "c")

    def test_actual_usage_aggregation_and_hit_zeroes(self):
        class Fake:
            model, max_tokens, last = "fake", 100, {}
            def __call__(self, q, c):
                self.last = {"usage": {"prompt_tokens": 100, "completion_tokens": 10},
                             "api_latency_seconds": 0.1, "finish_reason": "stop"}
                return "不开放"
        rows = load_data(ROOT / "data" / "toy_qa.jsonl")[:2]
        with tempfile.TemporaryDirectory() as directory:
            result = live_experiment(rows, Fake(), 0.5, Path(directory))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["runs"][0]["prompt_tokens"], 200)
            self.assertEqual(result["runs"][-1]["prompt_tokens"], 100)
            self.assertEqual(result["runs"][-1]["completion_tokens"], 10)
            with self.assertRaises(ValueError):
                live_experiment(rows, Fake(), 0.5, Path(directory))

    def test_failed_run_is_marked_and_partial_log_survives(self):
        class Fake:
            model, max_tokens, last, calls = "fake", 100, {}, 0
            def __call__(self, q, c):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated error")
                self.last = {"usage": {"prompt_tokens": 100, "completion_tokens": 10}}
                return "不开放"
        rows = load_data(ROOT / "data" / "toy_qa.jsonl")[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                live_experiment(rows, Fake(), 0.5, output)
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "incomplete")
            self.assertEqual(len((output / "requests.jsonl").read_text(encoding="utf-8").splitlines()), 1)
