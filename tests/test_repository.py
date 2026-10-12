import json
import re
import unittest
from pathlib import Path


class RepositoryTests(unittest.TestCase):
    def test_local_document_links_and_current_result_accounting(self):
        root = Path(__file__).resolve().parents[1]
        for document in [root/'README.md', *(root/'docs').glob('*.md'), *(root/'results/current').glob('*.md')]:
            for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
                if target.startswith(('http://','https://','#')):
                    continue
                self.assertTrue((document.parent/target.split('#',1)[0]).exists(), f'{document}: {target}')
        results = json.loads((root/'results/current/metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(sum(r['model_calls'] for r in results['runs']), results['actual_api_calls'])
        self.assertEqual([r['mode'] for r in results['runs']], results['arms'])
        self.assertTrue(all(r['checkpoints']==results['case_count'] for r in results['runs']))
