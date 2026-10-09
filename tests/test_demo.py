import io
import json
import unittest
from unittest.mock import Mock
from token_budget_lab.demo import Handler


class DemoTests(unittest.TestCase):
    def handler(self, payload):
        raw=json.dumps(payload).encode()
        handler=object.__new__(Handler)
        handler.headers={'Content-Length':str(len(raw))}
        handler.rfile=io.BytesIO(raw)
        handler.wfile=io.BytesIO()
        handler.send_response=Mock()
        handler.send_header=Mock()
        handler.end_headers=Mock()
        return handler

    def test_request_returns_budgeted_context(self):
        handler=self.handler(dict(query='Mercury',context='Earth has one moon. Mercury has none.',method='bm25_neighbor',ratio=.5))
        handler.do_POST()
        result=json.loads(handler.wfile.getvalue())
        self.assertLessEqual(len(result['selected']),result['budget'])
        handler.send_response.assert_called_once_with(200)

    def test_invalid_budget_returns_client_error(self):
        handler=self.handler(dict(query='q',context='c',method='bm25',ratio=-1))
        handler.do_POST()
        handler.send_response.assert_called_once_with(400)
        self.assertIn('error',json.loads(handler.wfile.getvalue()))
