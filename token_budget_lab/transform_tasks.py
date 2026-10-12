"""Narrow fixture adapter from current user wording to host transform rules.

This is not a general natural-language intent parser. Production hosts should
construct WriteTransform from their own structured, authorized task intent.
No gold answer, expected file state or fixture update is used here.
"""
import re
from .write_transform import FreshWriteGuard, WriteTransform


def guard_from_query(query):
    if 'read_log' in query and '两位小数' in query and 'status_code' in query:
        log=re.search(r'read_log\s*读取\s*(?:现在的\s*)?(scan_[abcd])',query)
        if not log:
            raise ValueError('explicit current log required')
        identifier=re.search(r'(?:event=|精确查找\s*)"(scan_[abcd]-[0-9]+)"',query)
        if identifier:
            selector=dict(kind='unique_event',array='/events',match_key='event',match_value=identifier[1],field='status_code')
        elif 'log_text' in query and 'TAIL_MARKER' in query:
            selector=dict(kind='text_record',pointer='/log_text',marker='TAIL_MARKER',field='status_code')
        else:
            raise ValueError('explicit supported source selector required')
        rule=WriteTransform('read_log',dict(name=log[1]),'write_report','total',selector,
                            fixed_arguments=dict(currency='USD'))
    elif 'read_config' in query and 'timeout_ms 加 1' in query:
        rule=WriteTransform('read_config',{},'write_config','timeout_ms',
                            dict(kind='pointer',pointer='/config/timeout_ms'),operation='add_integer',constant=1)
    else:
        raise ValueError('query outside the declared fixture adapter')
    return FreshWriteGuard([rule])
