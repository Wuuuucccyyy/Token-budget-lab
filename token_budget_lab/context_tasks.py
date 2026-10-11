"""New frozen tool fixtures for offload, temporal writes and missing evidence."""
import copy
import random

from .long_context_tasks import build_long_tasks


def build_context_tasks():
    rng = random.Random(20261012)
    episodes = []
    originals = build_long_tasks()['episodes']
    for kind in ('knowledge_update','temporal','abstention','tool_execution'):
        for n in range(2):
            original = next(e for e in originals if e['kind'] == kind and e['context_length'] == 'long' and e['id'].split('-')[1] == f'{n:02d}')
            e = copy.deepcopy(original)
            first, latest, quota = rng.randint(201,499), rng.randint(1301,1700), rng.randint(41,70)
            e.update(id=f'context-{kind}-{n:02d}',base_id=f'context-{kind}-{n:02d}',split='test',log_events=60)
            e['initial_config'].update(timeout_ms=first,quota=quota)
            for action in e['setup']:
                if action.get('change') == 'config' and action.get('module') == 'Alpha':
                    action.update(timeout_ms=latest,quota=quota)
            if kind == 'knowledge_update':
                e['checkpoints'][0].update(answer=str(latest+quota),timeout_ms=latest+quota)
                e['checkpoints'][1].update(answer=str(first+5),timeout_ms=first+5)
                e['checkpoints'][2].update(timeout_ms=first+5)
            elif kind == 'temporal':
                e['checkpoints'][0].update(answer=str(latest-first),timeout_ms=latest-first)
                e['checkpoints'][1].update(answer=str(latest),timeout_ms=latest)
                e['checkpoints'][2].update(timeout_ms=latest)
            elif kind == 'abstention':
                for cp in e['checkpoints']:cp['timeout_ms']=latest
                email=f'context-alpha-{n}@example.invalid'
                e['checkpoints'][2].update(query=f'现在明确补充并授权：Alpha 的 billing_email={email}。把它作为 recipient 写入本地 report.json，再执行 run_checks；answer 填该邮箱。不发送邮件。',
                    answer=email,report=dict(recipient=email))
            else:
                # Necessary facts INSIDE a bulky current tool result exercise
                # direct JSON-pointer restoration, not only irrelevant deletion.
                event=37+2*n
                code=[200,204,429,503][(event+2)%4]
                e['checkpoints']=[
                    dict(query=f'实际调用 read_log 读取 scan_c。在 events 数组查找 event="scan_c-{event}" 的 status_code。'
                         '将代码转为两位小数字符串作为 report.json 的 total，currency 用 USD，这是日志指标记录；实际运行 run_checks。'
                         '最后 answer 只填两位小数。不改 config，不发送邮件。',answer=f'{code:.2f}',timeout_ms=latest,
                         report=dict(total=f'{code:.2f}',currency='USD'),required=['read_log','write_report','run_checks']),
                    dict(query='更正：report.json 的 currency 改为 CNY，total 保持刚才已写入的值。更新本地报告并运行 run_checks，answer 仍填两位小数。',
                         answer=f'{code:.2f}',timeout_ms=latest,report=dict(total=f'{code:.2f}',currency='CNY'),required=['write_report','run_checks']),
                    dict(query='汇报 report.json 中的 total，answer 填两位小数；不修改文件。',answer=f'{code:.2f}',timeout_ms=latest,
                         report=dict(total=f'{code:.2f}',currency='CNY'),required=[],no_writes=True)]
            episodes.append(e)
    return dict(schema_version='context-fixtures-v1',seed=20261012,
        source='original synthetic fixtures; new instances of existing task templates plus current-log evidence tasks; not a natural holdout benchmark',
        episodes=episodes)
