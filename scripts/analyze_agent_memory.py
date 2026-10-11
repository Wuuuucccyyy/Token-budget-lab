"""Reproduce a post-hoc format diagnostic without changing primary scores."""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from token_budget_lab.metrics import normalize, score_answer


def read_literal(raw):
    """Accept a whole JSON answer or one bare scalar; never mine prose for numbers."""
    text = raw.strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict) and set(value) == {'answer'} and isinstance(value['answer'], str):
        return value['answer'], 'strict_json'
    if isinstance(value, str):
        return value, 'json_string'
    if re.fullmatch(r'-?\d+|[A-Za-z][A-Za-z0-9_]*', text):
        return text, 'bare_scalar'
    return None, 'needs_review'


def analyze(answer_rows, data):
    references = {q['id']: q['answers'] for s in data['sessions'] for q in s['questions']}
    rows = []
    for row in answer_rows:
        value, kind = read_literal(row['raw_answer'])
        metric = score_answer(value, references[row['id']]) if value is not None else None
        rows.append(dict(id=row['id'], mode=row['mode'], format=kind, parsed_answer=value,
                         literal_exact_match=None if metric is None else metric['exact_match']))
    full = {r['id']: r for r in rows if r['mode'] == 'full'}
    groups = defaultdict(list)
    for row in rows:
        groups[row['mode']].append(row)
    summary = dict(status='post_hoc_format_diagnostic_not_primary_or_semantic_certification', runs=[])
    for mode in ('full', 'compact', 'recover'):
        rs = groups[mode]
        comparable = [r for r in rs if r['parsed_answer'] is not None and full[r['id']]['parsed_answer'] is not None]
        summary['runs'].append(dict(mode=mode, questions=len(rs),
            strict_json=sum(r['format']=='strict_json' for r in rs),
            bare_or_string=sum(r['format'] in {'bare_scalar','json_string'} for r in rs),
            needs_review=sum(r['parsed_answer'] is None for r in rs),
            literal_correct=sum(r['literal_exact_match']==1 for r in rs),
            literal_incorrect=sum(r['literal_exact_match']==0 for r in rs),
            comparable_with_full=len(comparable),
            literal_same_as_full=sum(normalize(r['parsed_answer'])==normalize(full[r['id']]['parsed_answer']) for r in comparable)))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--journal', type=Path, help='Local request journal; export final answers from synthetic fixtures only')
    source.add_argument('--answers', type=Path, help='Published final_answers.csv; requires no API')
    parser.add_argument('--data', type=Path, default=ROOT/'data/agent_memory.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.journal:
        records = [json.loads(s) for s in args.journal.read_text(encoding='utf-8').splitlines()]
        last = {}
        for r in records:
            if r['event']=='response':
                last[(r['id'], r['mode'])] = r.get('message', {}).get('content') or ''
        answers = [dict(id=r['id'], mode=r['mode'], raw_answer=last.get((r['id'],r['mode']),''))
                   for r in records if r['event']=='task_result']
        with (args.output/'final_answers.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['id','mode','raw_answer'],lineterminator='\n')
            writer.writeheader()
            writer.writerows(answers)
    else:
        with args.answers.open(encoding='utf-8',newline='') as handle:
            answers=list(csv.DictReader(handle))
    rows,summary=analyze(answers,json.loads(args.data.read_text(encoding='utf-8')))
    (args.output/'format_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (args.output/'format_diagnostics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
