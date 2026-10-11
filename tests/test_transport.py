import io
import json
import os
import unittest
from unittest.mock import patch

from token_budget_lab import DeepSeekChat, run_agent
from token_budget_lab.deepseek import normalize_usage, NoRedirect
from token_budget_lab.tool_tasks import tool
from helpers import Script


class TransportTests(unittest.TestCase):
    def test_chat_payload_uses_tools_and_preserves_reported_cache_usage(self):
        payload = dict(model='test', usage=dict(prompt_tokens=100, completion_tokens=5,
            prompt_cache_hit_tokens=80, prompt_cache_miss_tokens=20),
            choices=[dict(finish_reason='stop', message=dict(role='assistant', content='done'))])
        captured = []
        class Opener:
            def open(self, request, timeout):
                captured.append(json.loads(request.data))
                return io.BytesIO(json.dumps(payload).encode())
        with patch.dict(os.environ, DEEPSEEK_API_KEY='local-test-value'), patch('urllib.request.build_opener', return_value=Opener()):
            client = DeepSeekChat(model='test', json_output=True)
            response = client([dict(role='user', content='Finish.')], [tool('read', 'Read.')])
        self.assertEqual(captured[0]['tools'][0]['function']['name'], 'read')
        self.assertEqual(response['usage']['prompt_cache_hit_tokens'], 80)
        self.assertEqual(response['message']['content'], 'done')

    def test_malformed_reply_retains_known_usage_without_exposing_key(self):
        payload = dict(usage=dict(prompt_tokens=19, completion_tokens=3), choices=[])
        class Opener:
            def open(self, request, timeout):
                return io.BytesIO(json.dumps(payload).encode())
        with patch.dict(os.environ, DEEPSEEK_API_KEY='private-test-value'), patch('urllib.request.build_opener', return_value=Opener()):
            result = run_agent([dict(role='user', content='Finish.')], DeepSeekChat(), None,
                               't', 's', mode='full', memory_tools=False)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['prompt_tokens'], 19)
        self.assertEqual(result['completion_tokens'], 3)
        self.assertNotIn('private-test-value', str(result))

    def test_truncation_never_executes_requested_business_tool(self):
        class Truncated(Script):
            def __call__(self, messages, tools):
                value = super().__call__(messages, tools)
                value['finish_reason'] = 'length'
                return value
        actual = []
        result = run_agent([dict(role='user', content='Read.')], Truncated([[('read', {})]]), None,
            't', 's', mode='full', memory_tools=False, extra_tools=[tool('read', 'Read.')],
            handlers={'read': lambda: actual.append(True)})
        self.assertEqual(result['status'], 'truncated')
        self.assertEqual(actual, [])
        self.assertEqual(result['api_calls'], 1)

    def test_paid_response_is_persisted_before_handler_mutation(self):
        order = []
        run_agent([dict(role='user', content='Read.')], Script([[('read', {})], 'done']), None,
            't', 's', mode='full', memory_tools=False, extra_tools=[tool('read', 'Read.')],
            handlers={'read':lambda:order.append('handler')}, after_response=lambda r:order.append('response'))
        self.assertEqual(order, ['response', 'handler', 'response'])

    def test_invalid_usage_and_redirect_are_rejected(self):
        for row in ({}, dict(prompt_tokens=True, completion_tokens=1), dict(prompt_tokens=1, completion_tokens=-1)):
            with self.assertRaises(RuntimeError):
                normalize_usage(row, 'chat')
        with self.assertRaises(RuntimeError):
            NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid')
