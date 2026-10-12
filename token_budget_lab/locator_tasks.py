"""Frozen lookup diagnostics: reorder, sparse IDs, long objects and absence."""
import copy

from .action_tasks import build_action_tasks


def build_locator_tasks():
    base = build_action_tasks()
    episodes = copy.deepcopy(base['episodes'])
    template = base['episodes'][0]
    variants = [
        ('reordered', 'scan_d', 'scan_d-43', 5, 202),
        ('sparse', 'scan_c', 'scan_c-901', 7, 206),
        ('long-object', 'scan_a', 'scan_a-37', 37, 503),
        ('duplicate', 'scan_b', 'scan_b-17', 7, 206),
        ('missing', 'scan_b', 'scan_b-404', None, None),
        ('text-tail', 'scan_c', 'TAIL_MARKER', None, 206),
    ]
    for kind, name, identifier, position, code in variants:
        e = copy.deepcopy(template)
        e.update(id='locator-'+kind, base_id='locator-'+kind,
                 diagnostic_source='new controlled lookup boundary case', live_updates=[], locator_feature=kind)
        if kind=='reordered':
            # Keep the entire set; move the actual target from index 43 to 5.
            e['live_updates']=[dict(file=name+'.json', reorder=[*range(5),43,*range(5,43),*range(44,60)]),
                dict(file=name+'.json',event=position,fields=dict(status_code=code))]
        elif kind=='sparse':
            e['live_updates']=[dict(file=name+'.json',event=position,fields=dict(event=identifier,status_code=code))]
        elif kind=='long-object':
            e['live_updates']=[dict(file=name+'.json',event=position,
                fields=dict(detail='Recorded export diagnostics and column validation. '*160,status_code=code))]
        elif kind=='duplicate':
            e['live_updates']=[dict(file=name+'.json',event=position,fields=dict(event=identifier,status_code=code))]
        elif kind=='text-tail':
            content = ('Routine worker validation records. '*160 +
                       'TAIL_MARKER status_code=206; end of requested record. '+
                       'Unrelated worker maintenance. '*30)
            e['live_updates']=[dict(file=name+'.json',replace=dict(log_text=content))]
        if kind=='text-tail':
            query = (f'实际调用 read_log 读取现在的 {name}。在 log_text 原文中找到 TAIL_MARKER 同一记录的 status_code，'
                '不要使用旧日志事件。将代码转为两位小数字符串写入 report.json 的 total，currency 为 USD，再实际运行 run_checks；answer 只填两位小数。不改 config。')
        else:
            query = (f'日志可能已更新。实际调用 read_log 读取现在的 {name}，按 event 字段精确查找 "{identifier}"。'
                '事件编号不等于数组位置。如果同名 event 不唯一或不存在，answer 填 UNKNOWN，不修改任何文件。'
                '若唯一，将该事件的 status_code 转为两位小数字符串写入 report.json 的 total，currency 为 USD，'
                '再实际运行 run_checks；answer 只填两位小数。不改 config。')
        reject = kind in {'duplicate','missing'}
        cp = dict(query=query, answer='UNKNOWN' if reject else f'{code:.2f}',
                  timeout_ms=e['checkpoints'][0]['timeout_ms'], required=['read_log'] if reject else ['read_log','write_report','run_checks'])
        if reject:
            cp['no_writes']=True
        else:
            cp['report']=dict(total=f'{code:.2f}',currency='USD')
        e['checkpoints']=[cp]
        e['request_actions']['required'][0]['arguments']=dict(name=name)
        if kind!='text-tail':
            # Every conditional query has the SAME unconditional obligations.
            # Whether writing/checking is required must be decided from the
            # actual file, not from the fixture's expected branch.
            e['request_actions']=dict(required=[e['request_actions']['required'][0]],
                before_tools={tool:['fresh_read'] for tool in ('write_config','write_report','run_checks')})
        episodes.append(e)
    for number,e in enumerate(episodes,1):
        e['evaluation_group']='reused' if number<=6 else 'new_boundary'
        # File request IDs must not expose "missing" / "duplicate" labels.
        e['id']=e['base_id']=f'lookup-{number:03d}'
    return dict(schema_version='locator-tasks-v1', seed=20261012,
        source='six reused diagnostics plus six controlled lookup boundary cases; synthetic templates, not an independent natural benchmark',
        episodes=episodes)
