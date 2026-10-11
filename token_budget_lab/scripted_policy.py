"""Fixture-specific offline driver: reads visible messages, never reference labels.

This exercises file tools and recovery wiring. Its results are NOT LLM accuracy.
"""
import json
import re
from .memory import encode


class ScriptedPolicy:
    def __init__(self):
        self.last={}
        self.calls=0

    def __call__(self,messages,tools):
        self.calls+=1
        query_index=max(i for i,m in enumerate(messages) if m['role']=='user')
        query=messages[query_index]['content']
        names={t['function']['name'] for t in tools}
        records=[]
        current=[]
        for i,m in enumerate(messages):
            if m['role']!='tool': continue
            try: value=json.loads(m['content'])
            except (ValueError,TypeError): continue
            if isinstance(value,dict):
                records.append(value)
                if i>query_index: current.append(value)

        def call(name,**arguments):
            return dict(message=dict(role='assistant',content=None,tool_calls=[dict(
                id=f'policy-{self.calls}',type='function',function=dict(name=name,arguments=encode(arguments)))]),finish_reason='tool_calls')

        def answer(value):
            return dict(message=dict(role='assistant',content=encode(dict(answer=str(value)))),finish_reason='stop')

        def missing():
            return call('history_restore_all') if 'history_restore_all' in names and not any(r.get('full_history') for r in current) else answer('UNKNOWN')

        def finish_after_checks(value):
            checks=[r for r in current if 'passed' in r]
            return answer(value) if checks else call('run_checks')

        email=re.search(r'billing_email=([^\s。]+)',query)
        if email:
            if not any(r.get('written') and 'report' in r for r in current):
                return call('write_report',recipient=email.group(1))
            return finish_after_checks(email.group(1))
        if 'billing_email' in query:
            # Other-module log contacts are never a source for Alpha.
            configs=[r['config'] for r in records if r.get('config',{}).get('module')=='Alpha']
            return answer(configs[-1].get('billing_email','UNKNOWN')) if configs else call('read_config')
        if '汇报最近一次 run_checks' in query:
            checks=[r for r in records if 'passed' in r]
            return answer('PASS' if checks[-1]['passed'] else 'FAIL') if checks else missing()
        if 'timeout_ms' in query:
            snapshots=[r for r in records if 'observed_at' in r and r.get('config',{}).get('module')=='Alpha']
            date=re.search(r'截至 (\d{4}-\d{2}-\d{2})',query)
            if date: snapshots=[r for r in snapshots if r['observed_at']<=date.group(1)]
            if not snapshots: return missing()
            snapshots.sort(key=lambda r:r['observed_at'])
            selected=snapshots[0] if '首次' in query else snapshots[-1]
            value=selected['config']['timeout_ms']+(5 if '加 5' in query else 0)
            if not any(r.get('written') and 'config' in r for r in current):
                return call('write_config',timeout_ms=value)
            return finish_after_checks(value)
        if '最初读取的 orders' in query:
            original=[r for r in records if 'items' in r and 'observed_at' in r]
            if not original: return missing()
            totals=[r['total'] for r in current if 'total' in r]
            if not totals: return call('calculate_total',items=original[0]['items'])
            value=totals[-1]
            if not any(r.get('written') and 'report' in r for r in current):
                return call('write_report',total=value,currency='USD')
            return finish_after_checks(value)
        if 'report.json' in query:
            reports=[r['report'] for r in current if 'report' in r]
            if not reports: return call('read_report')
            if not reports[-1] or 'total' not in reports[-1]: return answer('UNKNOWN')
            value=reports[-1]['total']
            if 'CNY' in query:
                if not any(r.get('written') for r in current):
                    return call('write_report',total=value,currency='CNY')
                return finish_after_checks(value)
            return answer(value)
        return answer('UNKNOWN')
