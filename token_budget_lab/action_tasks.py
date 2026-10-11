"""Focused fresh-read diagnostics; structured obligations contain no answers."""
import copy

from .action_contract import ActionContract, RequiredAction
from .context_tasks import build_context_tasks


def build_action_tasks():
    originals = [e for e in build_context_tasks()['episodes'] if e['kind'] == 'tool_execution']
    episodes = []
    variants = [
        ('replay-00', 'scan_c', 37, None),
        ('replay-01', 'scan_c', 39, None),
        ('changed-log-00', 'scan_c', 37, 206),
        ('changed-log-01', 'scan_a', 15, 409),
        ('changed-log-02', 'scan_d', 43, 202),
        ('changed-config-00', None, None, 1837),
    ]
    for n, (identity, name, event, updated) in enumerate(variants):
        e = copy.deepcopy(originals[n % 2])
        e.update(id='actions-' + identity, base_id='actions-' + identity,
                 diagnostic_source='v0.6 task replay' if updated is None else 'new controlled current-file change',
                 live_updates=[])
        current_timeout = e['checkpoints'][0]['timeout_ms']
        if name:
            j = ['scan_a', 'scan_b', 'scan_c', 'scan_d'].index(name)
            code = [200, 204, 429, 503][(event + j) % 4] if updated is None else updated
            query = (f'实际调用 read_log 读取 {name}。在 events 数组查找 event="{name}-{event}" 的 status_code。'
                     '将代码转为两位小数字符串作为 report.json 的 total，currency 用 USD，这是日志指标记录；'
                     '实际运行 run_checks。最后 answer 只填两位小数。不改 config，不发送邮件。')
            if updated is not None:
                query = '文件在历史读取后已由外部更新，旧结果不能替代本轮读取。' + query
                e['live_updates'] = [dict(file=name + '.json', event=event,
                                          fields=dict(status_code=updated, observed_at='2026-10-12T10:00:00'))]
            cp = dict(query=query, answer=f'{code:.2f}', timeout_ms=current_timeout,
                      report=dict(total=f'{code:.2f}', currency='USD'),
                      required=['read_log', 'write_report', 'run_checks'])
            read_tool, read_args, write_tool = 'read_log', dict(name=name), 'write_report'
        else:
            e['live_updates'] = [dict(file='config.json', fields=dict(timeout_ms=updated, observed_at='2026-10-12'))]
            cp = dict(query='config.json 在历史读取后已由外部更新。实际调用 read_config 读取现在的 Alpha config，'
                      '把本轮读取的 timeout_ms 加 1 写入当前 config，保持 quota 不变，再实际调用 run_checks；'
                      'answer 只填写入后的数值。历史快照不能替代现在的读取，不生成报告。',
                      answer=str(updated + 1), timeout_ms=updated + 1,
                      required=['read_config', 'write_config', 'run_checks'])
            read_tool, read_args, write_tool = 'read_config', {}, 'write_config'
        e['checkpoints'] = [cp]
        # Structured user intent only. Numeric answers and expected file state
        # stay in the checkpoint's scoring fields and never enter this policy.
        e['request_actions'] = dict(required=[
            dict(id='fresh_read', tool=read_tool, arguments=read_args),
            dict(id='write', tool=write_tool, after=['fresh_read'], result_matches=dict(written=True)),
            dict(id='check', tool='run_checks', after=['write'], result_matches=dict(passed=True)),
        ], before_tools={'write_config': ['fresh_read'], 'write_report': ['fresh_read'], 'run_checks': ['write']})
        episodes.append(e)
    return dict(schema_version='fresh-actions-v1', seed=20261012,
                source='focused synthetic diagnostic: two v0.6 task replays plus four current-file-change cases; not independent benchmark',
                episodes=episodes)


def execution_contract(request_actions):
    return ActionContract([RequiredAction(**a) for a in request_actions['required']], request_actions.get('before_tools'))


def update_current_files(sandbox, updates):
    for update in updates:
        if update['file'] not in {'config.json', 'scan_a.json', 'scan_b.json', 'scan_c.json', 'scan_d.json'}:
            raise ValueError('unknown fixture file')
        value = sandbox._read(update['file'])
        target = value['events'][update['event']] if 'event' in update else value
        target.update(copy.deepcopy(update['fields']))
        sandbox._write(update['file'], value)
