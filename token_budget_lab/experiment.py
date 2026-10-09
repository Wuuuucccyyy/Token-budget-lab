"""Config-driven, append-only experiments with explicit resume and call limits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

from .core import CounterBackend, compress, extractive_answer, fingerprint, prompt, SYSTEM
from .datasets import digest, load_rows
from .metrics import evaluate_answer

ROOT = Path(__file__).resolve().parents[1]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hash():
    return stable_hash({p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def prepare(config):
    if config['backend'] not in {'offline', 'deepseek'}:
        raise ValueError('invalid backend')
    if not config.get('strategies'):
        raise ValueError('strategies required')
    if config.get('repeats', 1) < 1:
        raise ValueError('repeats must be positive')
    names = set()
    for strategy in config['strategies']:
        if strategy['name'] in names:
            raise ValueError('duplicate strategy name')
        names.add(strategy['name'])
        if strategy['method'] not in {'full', 'head', 'bm25', 'bm25_neighbor', 'llmlingua2'}:
            raise ValueError('invalid method')
        if strategy.get('cache', 'none') not in {'none', 'exact'}:
            raise ValueError('runner supports none/exact cache only')
        if not 0 < strategy.get('ratio', 1) <= 1 or strategy.get('radius', 1) < 0:
            raise ValueError('invalid compression parameters')
    path = ROOT / config['data']
    rows = load_rows(path, config.get('split'))
    if config.get('sampling') == 'document_round_robin':
        groups = {}
        for row in rows:
            groups.setdefault(row['document_id'], []).append(row)
        docs = sorted(groups)
        random.Random(config.get('seed', 20261010)).shuffle(docs)
        rows = [groups[d][i] for i in range(max(map(len, groups.values())))
                for d in docs if i < len(groups[d])]
    if 'limit' in config:
        if config['limit'] < 1:
            raise ValueError('limit must be positive')
        rows = rows[:config['limit']]
    if config['backend'] == 'deepseek':
        for key in ('model', 'protocol', 'thinking', 'max_tokens', 'max_calls'):
            if key not in config:
                raise ValueError(f'explicit {key} required for paid experiment')
        if config['max_calls'] < 1 or config['max_tokens'] < 1:
            raise ValueError('call/output budgets must be positive')
    manifest = dict(schema_version=1, config=config, data_sha256=digest(path),
                    code_sha256=code_hash(), question_ids=[r['id'] for r in rows],
                    python=platform.python_version(), budget_unit=config.get('encoding', 'characters'))
    manifest['identity'] = stable_hash({k: v for k, v in manifest.items() if k != 'python'})
    return rows, manifest


def plan(config):
    rows, manifest = prepare(config)
    calls = 0
    for strategy in config['strategies']:
        unique = {(r['namespace'], r['query'], r['context']) for r in rows}
        calls += len(unique) if strategy.get('cache') == 'exact' else len(rows)
    calls *= config.get('repeats', 1)
    lengths = sorted(len(r['context']) for r in rows)
    return dict(identity=manifest['identity'], questions=len(rows),
                documents=len({r.get('document_id', r['context']) for r in rows}),
                context_characters_min=lengths[0], context_characters_median=lengths[len(lengths)//2],
                context_characters_max=lengths[-1],
                planned_calls_without_failures=calls if config['backend'] == 'deepseek' else 0,
                max_calls=config.get('max_calls'),
                note='Characters are not provider tokens. API usage is known only after a response.')


def execute(config, output, client=None, resume=False):
    rows, manifest = prepare(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, log_path = output / 'manifest.json', output / 'requests.jsonl'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not resume:
            raise ValueError('existing experiment; use --resume')
        if old['identity'] != manifest['identity']:
            raise ValueError('config, data or code changed; use a new output directory')
    else:
        if log_path.exists():
            raise ValueError('orphan log without manifest')
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    records = [json.loads(s) for s in log_path.read_text(encoding='utf-8').splitlines()] if log_path.exists() else []
    # A pending request may have reached the provider. Never silently repeat it.
    settled = {r['key'] for r in records if r['event'] == 'result'}
    if any(r['event'] == 'start' and r['key'] not in settled for r in records):
        raise RuntimeError('unresolved API attempt; inspect provider billing/log before creating a new run')
    if any(r.get('status') == 'error' for r in records):
        raise RuntimeError('logged API error; no automatic paid retry, use a new run after inspection')
    completed = {r['key']: r for r in records if r['event'] == 'result'}
    attempts = sum(r['event'] == 'start' for r in records)
    counter = CounterBackend(config.get('encoding', 'characters'))
    llm_compressor = None
    if any(s['method'] == 'llmlingua2' for s in config['strategies']):
        from .llmlingua_adapter import LLMLingua2
        llm_compressor = LLMLingua2(**config['llmlingua2'])
    if config['backend'] == 'deepseek' and client is None:
        from .deepseek import DeepSeek
        client = DeepSeek(config['model'], config['max_tokens'], config.get('timeout', 120),
                          protocol=config['protocol'], thinking=config['thinking'], allow_empty=True)
    status = 'complete'
    with log_path.open('a', encoding='utf-8') as log:
        def append(record):
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
            records.append(record)
        rng = random.Random(config.get('seed', 20261010))
        try:
            for repeat in range(config.get('repeats', 1)):
                strategies = list(config['strategies'])
                rng.shuffle(strategies)
                for strategy in strategies:
                    cache = {}
                    for row in rows:
                        key = f"{repeat}:{strategy['name']}:{row['id']}"
                        cache_key = fingerprint(row['context'], row['namespace'], config.get('model', 'offline'),
                                                SYSTEM, stable_hash(strategy)) + stable_hash(row['query'])
                        if key in completed:
                            previous = completed[key]
                            if previous['status'] == 'ok' and strategy.get('cache') == 'exact':
                                cache[cache_key] = previous
                            continue
                        cached = cache.get(cache_key) if strategy.get('cache') == 'exact' else None
                        if not cached and config['backend'] == 'deepseek' and attempts >= config['max_calls']:
                            status = 'call_limit'
                            raise StopIteration
                        start = time.perf_counter()
                        budget = math.floor(counter.count(row['context']) * strategy.get('ratio', 1))
                        selected = row['context']
                        if cached:
                            selected = cached['selected_context']
                        elif strategy['method'] == 'llmlingua2':
                            selected = llm_compressor.compress(row['query'], row['context'], budget, counter)
                        elif strategy['method'] != 'full':
                            selected = compress(row['query'], row['context'], budget, counter,
                                                strategy['method'], strategy.get('radius', 1))
                        compression_seconds = time.perf_counter() - start
                        metadata = {}
                        record_status = 'ok'
                        if cached:
                            answer = cached['answer']
                            metadata = dict(usage={'prompt_tokens': 0, 'completion_tokens': 0}, finish_reason='cache')
                        elif config['backend'] == 'offline':
                            answer = extractive_answer(row['query'], selected)
                        else:
                            append(dict(event='start', key=key, strategy=strategy['name'], id=row['id']))
                            attempts += 1
                            client.last = {}
                            try:
                                answer = client(row['query'], selected)
                                metadata = dict(client.last)
                            except Exception:
                                append(dict(event='result', key=key, status='error', id=row['id'],
                                            strategy=strategy['name'], usage=client.last.get('usage'),
                                            note='Request failed; billing unknown unless usage is present. No automatic retry.'))
                                raise RuntimeError('API request failed; preserved available usage without credentials') from None
                            if not answer.strip():
                                record_status = 'empty'
                            elif metadata.get('finish_reason') in {'length', 'max_tokens'}:
                                record_status = 'truncated'
                        record = dict(event='result', key=key, id=row['id'], document_id=row.get('document_id', row['namespace']),
                                      strategy=strategy['name'], repeat=repeat, status=record_status,
                                      cache_hit=bool(cached), cache_source=cached['key'] if cached else None,
                                      answer=answer, selected_context=selected,
                                      original_context_units=counter.count(row['context']), context_budget=budget,
                                      selected_context_units=counter.count(selected),
                                      input_units=0 if cached else counter.count(prompt(row['query'], selected)),
                                      baseline_input_units=counter.count(prompt(row['query'], row['context'])),
                                      compression_seconds=compression_seconds, end_to_end_seconds=time.perf_counter()-start,
                                      **metadata, **evaluate_answer(row, answer, selected))
                        append(record)
                        completed[key] = record
                        if record_status == 'ok' and strategy.get('cache') == 'exact':
                            cache[cache_key] = record
        except StopIteration:
            pass
        except BaseException:
            status = 'incomplete'
            raise
        finally:
            from .analysis import summarize
            summary = summarize(records, manifest)
            summary['status'] = status
            (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.dry_run:
        print(json.dumps(plan(config), indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --dry-run')
    summary = execute(config, args.output, resume=args.resume)
    from .analysis import write_report
    write_report(summary, args.output)
    print(f"{summary['status']}: {args.output.resolve()}")


if __name__ == '__main__':
    main()
