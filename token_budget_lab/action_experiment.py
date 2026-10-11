"""Fixed six-step fresh-action diagnostic, including rejected final answers."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .action_tasks import execution_contract, update_current_files
from .cache_cost import cache_cost
from .context_plugin import ContextPlugin
from .experiment import ROOT, code_hash, stable_hash
from .memory import HistoryArchive, encode
from .memory_agent import DeepSeekChat, run_agent
from .tool_experiment import summarize
from .tool_tasks import TOOLS, ToolSandbox, assess


ARMS = ['full', 'full_contract', 'context', 'context_contract']


def plan(config):
    if config['arms'] != ARMS or config['max_steps'] != 6 or config['max_calls'] < 1:
        raise ValueError('four arms, original six-step cap and paid-call cap required')
    path = ROOT / config['data']
    bundle = json.loads(path.read_text(encoding='utf-8'))
    for e in bundle['episodes']:
        execution_contract(e['request_actions'])
    return bundle['episodes'], dict(
        config={**config, 'modes': ARMS, 'backend': 'deepseek'},
        episode_ids=[e['id'] for e in bundle['episodes']], checkpoints=len(bundle['episodes']),
        expected_episode_checkpoints={e['id']: 1 for e in bundle['episodes']},
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), code_sha256=code_hash(),
        dataset_source=bundle['source'], contract_source='host structured current-user intent; no automatic natural-language parsing; no answer/state labels',
        quality='six controlled long-history cases; no general near-lossless confirmation',
        cache_control='independent 128-hex system prefix per task/arm; natural within-task cache; cannot clear provider cache',
        baseline='full_contract and context_contract use the same execution contract; unguarded arms retained',
        protocol='chat', thinking='disabled', json_output=True,
        pricing_source='https://api-docs.deepseek.com/quick_start/pricing/', pricing_verified_date='2026-10-12',
        pricing_interpretation='fixed reference rates for comparable repricing, not an account invoice; no warm probes')


def execute(config, output, transport=None):
    episodes, manifest = plan(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() or (output / 'requests.jsonl').exists():
        raise ValueError('fresh output required; no automatic paid replay')
    manifest.update(run_id=uuid.uuid4().hex, started_at_utc=datetime.now(timezone.utc).isoformat())
    manifest['identity'] = stable_hash(manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output / 'execution_sources').mkdir()
    for p in (ROOT / 'token_budget_lab').glob('*.py'):
        (output / 'execution_sources' / p.name).write_bytes(p.read_bytes())
    records, attempts = [], 0
    rng = random.Random(config['seed'])
    rng.shuffle(episodes)
    client = transport or DeepSeekChat(config['model'], config['max_tokens'], config.get('timeout', 120), json_output=True)
    archive = HistoryArchive(output / 'history.sqlite3')
    plugin = ContextPlugin(archive, budget=config['budget_characters'], recent_turns=1,
                           offload_tools=config['offload_tools'], offload_threshold=config['offload_threshold'])
    try:
        with (output / 'requests.jsonl').open('x', encoding='utf-8') as log:
            def append(row):
                log.write(json.dumps(row, ensure_ascii=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                records.append(row)
            for episode in episodes:
                arms = ARMS.copy()
                rng.shuffle(arms)
                for arm in arms:
                    active = [False]
                    meta = dict(episode_id=episode['id'], base_id=episode['base_id'], kind=episode['kind'],
                                context_length='long', mode=arm, checkpoint=0)
                    def action(row):
                        if active[0]:
                            append(dict(event='business_tool', **meta, **row))
                    sandbox = ToolSandbox(output / 'sandboxes' / episode['id'] / arm, episode, action)
                    history, _ = sandbox.seed_history(episode)
                    update_current_files(sandbox, episode['live_updates'])
                    history[0]['content'] = ('[Benchmark namespace; no task facts] '+
                        hashlib.sha512(encode([manifest['run_id'], episode['id'], arm]).encode()).hexdigest()+'\n'+history[0]['content'])
                    checkpoint = episode['checkpoints'][0]
                    history.append(dict(role='user', content=checkpoint['query']))
                    policy = execution_contract(episode['request_actions']) if arm.endswith('_contract') else None
                    trace_start, hashes = len(sandbox.trace), sandbox.file_hashes()
                    initial, previous = [], []
                    active[0] = True
                    def before(step, p):
                        nonlocal attempts, previous
                        if attempts >= config['max_calls']:
                            raise RuntimeError('paid call limit reached')
                        common = 0
                        for a, b in zip(previous, p.messages):
                            if a != b:
                                break
                            common += 1
                        p.context_metadata['actual_common_prefix_messages'] = common
                        p.context_metadata['previous_actual_messages'] = len(previous)
                        previous = copy.deepcopy(p.messages)
                        append(dict(event='start', **meta, step=step, sent_messages_sha256=stable_hash(p.messages)))
                        attempts += 1
                        if step == 0:
                            initial.append(p)
                    def response(row):
                        append(dict(event='response', **meta, **row))
                    def tools(row):
                        append(dict(event='execution_controller', **meta, **row))
                    result = run_agent(history, client, plugin if arm.startswith('context') else None,
                        'actions', episode['id']+':'+arm, mode='recover' if arm.startswith('context') else 'full',
                        memory_tools=arm.startswith('context'), max_steps=config['max_steps'], max_recovery_rounds=3,
                        extra_tools=TOOLS, handlers=sandbox.handlers(), snapshot_read_tools=config['snapshot_read_tools'],
                        action_contract=policy, before_request=before, after_response=response, after_tools=tools)
                    score = assess(checkpoint, result, sandbox, hashes, trace_start, episode['initial_config'])
                    required_read = episode['request_actions']['required'][0]
                    fresh = any(a['success'] and a['tool'] == required_read['tool'] and a['arguments'] == required_read['arguments']
                                for a in sandbox.trace[trace_start:])
                    score['required_tools_completed'] = score['required_tools_completed'] and fresh
                    score['task_success'] = score['task_success'] and fresh
                    append(dict(event='task_result', **meta, status=result['status'], **score,
                        fresh_read_completed=fresh, diagnostic_source=episode['diagnostic_source'],
                        execution_contract=result['action_contract'],
                        initial_history_mode=initial[0].mode, initial_history_reason=initial[0].reason,
                        initial_sent_units=initial[0].sent_units, initial_baseline_units=initial[0].baseline_units,
                        source_messages_retained=None, compressed_requests=sum(s['history_mode']=='compressed' for s in result['steps']),
                        fallback_requests=sum(s['history_mode']=='fallback_full' for s in result['steps']),
                        prompt_tokens=result['prompt_tokens'], completion_tokens=result['completion_tokens'],
                        model_calls=result['api_calls'], unknown_usage_requests=result['unknown_usage_requests'],
                        end_to_end_seconds=result['end_to_end_seconds']))
                    if result['status'] == 'error':
                        raise RuntimeError('API failure retained; no retry')
    finally:
        archive.close()
        summary = summarize(records, manifest)
        for run in summary['runs']:
            usages = [r['usage'] for r in records if r['event']=='response' and r['mode']==run['mode'] and r.get('usage') is not None]
            run.update(cache_cost(usages, config['usd_per_million']))
            if run['unknown_usage_requests']:
                run['estimated_usd'] = None
            tasks = [r for r in records if r['event']=='task_result' and r['mode']==run['mode']]
            run['fresh_read_completed'] = sum(r['fresh_read_completed'] for r in tasks)
        summary['total_api_attempts'] = attempts
        summary['research_usage'] = cache_cost([r.get('usage') or {} for r in records if r['event']=='response'], config['usd_per_million'])
        known = sum(r['event']=='response' and r.get('usage') is not None for r in records)
        summary['unknown_usage_attempts'] = attempts-known
        if attempts != known:
            summary['research_usage']['estimated_usd'] = None
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--plan', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.plan:
        print(json.dumps(plan(config)[1], ensure_ascii=True, indent=2))
        return
    if not args.output:
        p.error('--output required for paid execution')
    execute(config, args.output)


if __name__ == '__main__':
    main()
