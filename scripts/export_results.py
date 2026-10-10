"""Publish compact, credential-free artifacts from the public offline run."""
import argparse
import csv
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
if manifest['config']['backend'] != 'offline' or manifest['config']['data'] != 'data/squad_article.json':
    raise ValueError('This exporter is restricted to the public offline dataset')
args.output.mkdir(parents=True, exist_ok=True)
for name in ['manifest.json', 'summary.json', 'REPORT.md', 'FAILURE_CASES.md', 'tradeoff.png']:
    source = args.run / name
    if source.exists():
        shutil.copyfile(source, args.output / name)
fields = ['id', 'document_id', 'strategy', 'repeat', 'status', 'original_context_units',
          'context_budget', 'selected_context_units', 'input_units', 'baseline_input_units',
          'exact_match', 'f1', 'answer_contains_gold', 'answer_span_retained']
fields += ['timing_scope', 'cache_lookup_seconds', 'compression_seconds', 'end_to_end_seconds']
with (args.output / 'per_question.csv').open('w', encoding='utf-8', newline='') as target:
    writer = csv.DictWriter(target, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for line in (args.run / 'requests.jsonl').read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record['event'] == 'result':
            writer.writerow(record)
print(f'Published numerical results: {args.output}')
