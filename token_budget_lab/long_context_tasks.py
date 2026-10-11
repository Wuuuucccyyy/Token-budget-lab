"""Matched context-length conditions with multi-version, multi-source tasks."""
import copy
from .tool_tasks import build_tasks


def build_long_tasks():
    episodes = []
    for original in build_tasks()['episodes']:
        n = int(original['id'].rsplit('-', 1)[1])
        for length, versions, events in [('short', 0, 0), ('medium', 6, 12), ('long', 18, 48)]:
            e = copy.deepcopy(original)
            e.update(id=original['id']+'-'+length, base_id=original['id'], context_length=length,
                     log_events=events, background_profile='operational_records')
            # Source observations are deliberately separated by unrelated work.
            setup = [dict(tool='read_config', arguments={})]
            if e['kind'] == 'tool_execution':
                setup += [dict(tool='read_orders', arguments={}),
                          dict(change='orders', items=[dict(quantity=1, price='99.00')])]
            for j in range(versions):
                module = ['Beta', 'Gamma', 'Delta'][j % 3]
                setup += [dict(change='config', module=module, timeout_ms=1100+19*j+n,
                               quota=40+j, observed_at=f'2026-09-{10+j:02d}'),
                          dict(tool='read_config', arguments={})]
                if j == versions//2:
                    setup += [dict(tool='read_log', arguments=dict(name='scan_a'))]
            latest = next(s['timeout_ms'] for s in original['setup'] if s.get('change') == 'config')
            setup += [dict(change='config', module='Alpha', timeout_ms=latest,
                           quota=e['initial_config']['quota'], observed_at='2026-10-05')]
            setup += [dict(tool='read_config', arguments={})]
            if events:
                setup += [dict(tool='read_log', arguments=dict(name=name)) for name in ('scan_b','scan_c','scan_d')]
            # End at a small genuine file read, rather than a large diagnostic dump.
            setup += [dict(tool='read_report', arguments={})]
            e['setup'] = setup
            first = e['initial_config']['timeout_ms']
            quota = e['initial_config']['quota']
            if e['kind'] == 'knowledge_update':
                e['checkpoints'][0].update(
                    query='需要根据历史记录完成配置更新：仅针对 Alpha 模块，从最近一次读取的 config 快照取 timeout_ms 和 quota，'
                          '将两者相加作为新的 timeout_ms。其他模块和诊断日志只作为背景，不能混用。只改当前配置的 timeout_ms，'
                          '保持 quota 和其他字段不变，实际运行 run_checks；最后 answer 只填新的数值。',
                    answer=str(latest+quota), timeout_ms=latest+quota)
                e['checkpoints'][1].update(
                    query='撤回前一步的参数选择。使用首次读取的 Alpha config 快照中的 timeout_ms 加 5，'
                          '不要使用后来的写入结果，也不要沿用其他模块的数值。写入当前配置，保留 quota，实际运行 run_checks；answer 只填新数值。',
                    answer=str(first+5), timeout_ms=first+5)
            elif e['kind'] == 'temporal':
                e['checkpoints'][0].update(
                    query='比较两个历史时间点：截至 2026-10-04 与截至 2026-10-06，各选当时最近的 Alpha config 快照，'
                          '计算后一个 timeout_ms 减前一个 timeout_ms 的差值。不得用现在的文件值代替历史快照，其他模块不参与。'
                          '将差值写入当前 timeout_ms，保留 quota，实际运行 run_checks；answer 只填差值。',
                    answer=str(latest-first), timeout_ms=latest-first)
                e['checkpoints'][1]['query'] = ('恢复历史配置：截至 2026-10-06，选当时最近的 Alpha config 快照，'
                    '取该快照的 timeout_ms 写入当前配置。前一步的计算与写入发生在后来，不能作为这个时间点的证据。'
                    '保留 quota 并实际运行 run_checks；answer 只填数值。')
            elif e['kind'] == 'abstention':
                e['checkpoints'][0]['query'] = ('核对 Alpha 的 billing_email 是否有明确记录。历史包含多个模块和多次版本读取，'
                    '其他模块的联系人、日志路径和名称都不能代替 Alpha 的邮箱；不允许根据命名推断。'
                    '若没有明确记录，answer 填 UNKNOWN；本阶段不生成报告、不改配置，也不发送邮件。')
                e['checkpoints'][1]['query'] = ('再次检查：能否只根据 Alpha 模块名称推断 billing_email？'
                    '必须以明确记录为依据，禁止猜测或借用其他模块联系人。没有明确记录仍回答 UNKNOWN，不生成报告、不修改配置。')
            else:
                e['checkpoints'][0]['query'] = ('使用最初读取的 orders CSV 原始记录计算 quantity×price 合计。'
                    '当前 CSV 已被替换，因此不能重新读取后用新订单替代旧订单；诊断日志里的计数也不是订单金额。'
                    '必须实际调用 calculate_total，随后把金额和 USD 写入本地 report.json，再实际运行 run_checks。'
                    'answer 只填两位小数金额，不发送邮件，不改 config。')
            # Later checkpoints must reflect continuous state, including the new first write.
            episodes.append(e)
    return dict(schema_version='long-context-tools-v1', seed=20261011,
                source='original matched context-length fixtures; not official LongMemEval; scripted setup',
                design='24 base scenarios x 3 context lengths; dev/test isolated by base_id', episodes=episodes)
