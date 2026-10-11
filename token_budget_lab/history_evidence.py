"""Select original JSON tool records by explicit subject, fields and event time.

This planner reads only the query and canonical messages. It does not generate
facts, inspect reference labels, or certify that arbitrary prose is understood.
Unsupported or ambiguous demands return an explicit full-history fallback.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import date


def words(text):
    return set(re.findall(r'[a-z][a-z0-9_]*', text.lower()))


def keys(value):
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys(v)}
    return set()


@dataclass
class Record:
    turn_id: str
    position: int
    source: str
    entity: str | None
    observed_at: str | None
    fields: set
    raw: str


@dataclass
class EvidencePlan:
    supported: bool = False
    turn_ids: set = field(default_factory=set)
    reason: str = 'unstructured_query'
    details: dict = field(default_factory=dict)


def records_from_turns(turns):
    records = []
    for turn in turns:
        calls = {c['id']: c['function']['name'] for m in turn['messages']
                 for c in m.get('tool_calls', [])}
        for position, message in zip(turn['positions'], turn['messages']):
            name = calls.get(message.get('tool_call_id'), '')
            if message['role'] != 'tool' or name.startswith('history_'):
                continue
            try:
                value = json.loads(message.get('content', ''))
            except (ValueError, TypeError):
                continue
            if not isinstance(value, dict) or value.get('error'):
                continue
            source = re.sub(r'^(read|write|get|fetch)_', '', name)
            body = value.get(source, value)
            if not isinstance(body, dict):
                body = value
            entity = next((body[k] for k in ('module', 'entity', 'project', 'customer_id')
                           if isinstance(body.get(k), str)), None)
            observed = value.get('observed_at', body.get('observed_at'))
            if observed is not None:
                try:
                    observed = date.fromisoformat(observed[:10]).isoformat()
                except (ValueError, TypeError):
                    observed = None
            records.append(Record(turn['id'], position, source, entity, observed,
                                  {k.lower() for k in keys(value)}, message['content']))
    return records


def plan_evidence(query, turns):
    """Known tool schemas are inferred from records, never from task types/IDs."""
    records = records_from_turns(turns)
    terms = words(query)
    known_entities = {r.entity for r in records if r.entity}
    entities = {e for e in known_entities if e.lower() in terms}
    # A named entity explicitly rejected as a substitute is not a second demand.
    entities = {e for e in entities if not re.search(
        re.escape(e) + r'[^。；;]{0,24}(?:不能替代|不可替代|不得使用|不作为)', query, re.I)}
    sources = {r.source for r in records if r.source.lower() in terms}
    sources = {s for s in sources if any(not re.search(r'(?:不改|不修改|不得修改|禁止修改)\s*$',
        query[max(0,m.start()-12):m.start()]) for m in re.finditer(r'\b'+re.escape(s)+r'\b',query,re.I))}
    # A future tool execution is an action requirement, not a demand to restore
    # that tool's previous output (which may not have an observation date).
    sources = {s for s in sources if not re.search(r'(?:调用|执行|运行)\s*'+re.escape(s), query, re.I)}
    all_fields = {k for r in records for k in r.fields}
    source_names = {r.source for r in records}
    action_names = set(re.findall(r'(?:调用|执行|运行)\s*([A-Za-z][A-Za-z0-9_]*)', query))
    ignored = source_names | {t.lower() for t in action_names} | {'module','observed_at','answer','written','run_checks','read_config','write_config'}
    fields = ((terms & all_fields) | {t for t in terms if '_' in t}) - ignored
    if not sources and entities:
        # Prefer records with an explicit subject, rather than nested contacts in logs.
        sources = {r.source for r in records if r.entity in entities}
    if not sources:
        possible = {r.source for r in records if fields & r.fields}
        if len(possible) == 1:
            sources = possible
    if not sources or not (fields or entities or sources & terms):
        return EvidencePlan()
    candidates = [r for r in records if r.source in sources and
                  (not entities or r.entity in entities or r.entity is None)]
    details = dict(sources=sorted(sources), entities=sorted(entities), fields=sorted(fields))
    if not candidates:
        return EvidencePlan(reason='structured_subject_not_found', details=details)
    # Unrecognized temporal instructions must not silently become "latest".
    dates = list(dict.fromkeys(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', query)))
    for value in dates:
        try:
            date.fromisoformat(value)
        except ValueError:
            return EvidencePlan(reason='invalid_query_date', details=details)
    lower = query.lower()
    earliest = bool(re.search(r'首次|最初|最早|\b(first|earliest|initial)\b', lower))
    compare = bool(re.search(r'比较|差值|差额|变化|分别|\b(compare|difference|changed)\b', lower))
    broad = bool(re.search(r'所有|全部|历次|\b(all|every)\b', lower))
    as_of = bool(re.search(r'截至|截止|\bas of\b', lower))
    time_range = bool(re.search(r'期间|区间|之间|\bbetween\b', lower))
    if dates and not (as_of or time_range):
        return EvidencePlan(reason='unsupported_temporal_expression', details=details)
    missing = fields - {k for r in candidates for k in r.fields}
    # Missing structured fields cannot prove absence in unstructured history.
    # Fall back if other text actually mentions such a field with this subject.
    if missing:
        for turn in turns:
            text = '\n'.join(str(m.get('content') or '') for m in turn['messages']
                             if m['role'] in {'assistant', 'tool'})
            if words(text) & missing and (not entities or any(e.lower() in words(text) for e in entities)):
                return EvidencePlan(reason='unstructured_field_evidence', details=details)
    selected = []
    for source in sorted(sources):
        subject_groups = {}
        for r in candidates:
            if r.source == source:
                subject_groups.setdefault(r.entity, []).append(r)
        if not entities and len(subject_groups) > 1:
            return EvidencePlan(reason='ambiguous_subject', details=details)
        for group in subject_groups.values():
            # Missing dates are unsafe for date-specific selection.
            if dates and any(r.observed_at is None for r in group):
                return EvidencePlan(reason='undated_temporal_evidence', details=details)
            ordered = sorted(group, key=lambda r: (r.observed_at or '', r.position))
            if missing or broad or (compare and not dates):
                chosen = ordered
            elif dates and time_range and not as_of:
                if len(dates) != 2:
                    return EvidencePlan(reason='unsupported_time_range', details=details)
                start, end = sorted(dates)
                chosen = [r for r in ordered if start <= r.observed_at <= end]
            elif dates:
                chosen = []
                for cutoff in dates:
                    eligible = [r for r in ordered if r.observed_at <= cutoff]
                    if not eligible:
                        return EvidencePlan(reason='no_evidence_before_cutoff', details=details)
                    chosen.append(eligible[0] if earliest else eligible[-1])
            else:
                chosen = [ordered[0] if earliest else ordered[-1]]
            # Keep all conflicting records on the selected date. Sequence alone
            # is insufficient to assert that an equal-date conflict is resolved.
            for r in list(chosen):
                if r.observed_at:
                    chosen += [other for other in group if other.observed_at == r.observed_at and other not in chosen]
            selected += chosen
    if not selected:
        return EvidencePlan(reason='empty_temporal_evidence', details=details)
    details.update(temporal='as_of' if as_of else 'range' if time_range else
                   'all_versions' if broad or compare or missing else 'earliest' if earliest else 'latest',
                   cutoffs=dates, missing_fields=sorted(missing), records=len(selected))
    return EvidencePlan(True, {r.turn_id for r in selected}, 'structured_evidence_selected', details)
