"""Opt-in write validation against explicit snapshot requirements, never labels."""
from __future__ import annotations

import copy
import json
import re
from datetime import date

from .context_plugin import source_id, tool_sources
from .memory import encode


class ProvenanceRejected(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

    def result(self):
        return dict(error='provenance_rejected', reason=self.reason, written=False,
                    note='No write executed. Recheck the current task, original snapshot sources, and evidence declaration.')


class HistoricalWriteGuard:
    """Validate one declared numeric snapshot-write contract.

    Supported demands: explicit latest/first snapshots, one/two as-of dates,
    direct copy, two-field sum, second-minus-first, or a stated integer offset.
    observed_at means snapshot observation time, not universal fact validity.
    Unsupported/conflicting demands reject writes rather than guess.
    Hosts opt in with a business schema; unconfigured tools remain unchanged.
    """
    def __init__(self, messages, write_tool='write_config', read_tool='read_config',
                 value_argument='timeout_ms', root='config'):
        self.write_tool, self.read_tool = write_tool, read_tool
        self.value_argument, self.root = value_argument, root
        last_user = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
        self.query = messages[last_user].get('content', '') if last_user >= 0 else ''
        self.facts, self.undated_entities = {}, set()
        for sid, r in tool_sources(messages[:last_user]).items():
            if r['tool'] != read_tool:
                continue
            try:
                value = json.loads(r['message']['content'])
                body = value[root]
                if not isinstance(body, dict):
                    continue
                entity = body.get('module', body.get('entity'))
                try:
                    when = date.fromisoformat(value.get('observed_at', body.get('observed_at'))[:10]).isoformat()
                except (ValueError, TypeError):
                    if isinstance(entity, str):self.undated_entities.add(entity)
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            if not isinstance(body, dict) or not isinstance(entity, str):
                continue
            self.facts[sid] = dict(position=r['position'], entity=entity, observed_at=when, body=body)
        self.error = None
        try:
            self.demand = self._demand()
        except ProvenanceRejected as exc:
            self.demand, self.error = None, exc.reason

    def _demand(self):
        query = self.query
        if self.value_argument not in query:
            raise ProvenanceRejected('configured_field_not_requested_by_current_task')
        entities = {f['entity'] for f in self.facts.values() if re.search(r'\b'+re.escape(f['entity'])+r'\b', query, re.I)}
        if len(entities) != 1:
            raise ProvenanceRejected('explicit_single_subject_required')
        entity = next(iter(entities))
        if entity in self.undated_entities:
            raise ProvenanceRejected('undated_snapshot_cannot_establish_version_order')
        facts = {s:f for s,f in self.facts.items() if f['entity'] == entity}
        dates = list(dict.fromkeys(re.findall(r'\b\d{4}-\d{2}-\d{2}\b', query)))
        if dates and (len(dates) > 2 or not re.search(r'截至|截止|as of', query, re.I)):
            raise ProvenanceRejected('unsupported_time_requirement')
        for d in dates:
            try:
                date.fromisoformat(d)
            except ValueError:
                raise ProvenanceRejected('invalid_cutoff') from None
        first = bool(re.search(r'首次|最初|最早|\b(first|earliest|initial)\b', query, re.I))
        latest = bool(re.search(r'最近|最新|\b(latest|most recent)\b', query, re.I))
        if first and latest or not (dates or first or latest):
            raise ProvenanceRejected('unsupported_version_requirement')
        if dates and first:
            raise ProvenanceRejected('unsupported_first_as_of_combination')
        groups = []
        for cutoff in dates or [None]:
            eligible = {s:f for s,f in facts.items() if cutoff is None or f['observed_at'] <= cutoff}
            if not eligible:
                raise ProvenanceRejected('no_snapshot_for_requested_time')
            chosen_date = (min if first else max)(f['observed_at'] for f in eligible.values())
            group = {s:f for s,f in eligible.items() if f['observed_at'] == chosen_date}
            groups.append(list(group))
        fields = [self.value_argument]
        constant = 0
        if len(dates) == 2:
            if not re.search(r'后一个.*减前一个|second.*minus.*first', query, re.I):
                raise ProvenanceRejected('explicit_difference_order_required')
            operation = 'second_minus_first'
        elif re.search(r'相加|\bsum\b', query, re.I):
            numeric_keys = {k for f in facts.values() for k,v in f['body'].items() if type(v) is int}
            fields = sorted(k for k in numeric_keys if re.search(r'\b'+re.escape(k)+r'\b', query))
            if len(fields) != 2 or self.value_argument not in fields:
                raise ProvenanceRejected('explicit_two_numeric_fields_required')
            operation = 'sum'
        elif re.search(r'加\s*\d+|\bplus\s+\d+', query, re.I):
            match = re.search(r'(?:加|\bplus)\s*(\d+)', query, re.I)
            constant, operation = int(match.group(1)), 'add_constant'
        else:
            # Arithmetic wording outside this small contract must never fall
            # through to direct copy (multiply, divide, rounded prose, etc.).
            if re.search(r'减|乘|除|差值|差额|\b(minus|multiply|divide|difference)\b', query, re.I):
                raise ProvenanceRejected('unsupported_arithmetic_requirement')
            operation = 'copy'
        for group in groups:
            for field in fields:
                vals = [self.facts[s]['body'].get(field) for s in group]
                if any(type(v) is not int for v in vals) or len(set(vals)) != 1:
                    raise ProvenanceRejected('missing_or_conflicting_numeric_field')
        return dict(entity=entity, operation=operation, fields=fields, constant=constant,
                    source_groups=groups, cutoffs=dates)

    def notice(self):
        if not self.demand:
            return '[Write provenance contract] '+self.write_tool+' requires evidence; current demand is unsupported: '+str(self.error)+'.'
        sources = [{k:f[k] for k in ('entity', 'observed_at')} | dict(source_id=s, fields=self.demand['fields'])
                   for group in self.demand['source_groups'] for s in group for f in [self.facts[s]]]
        return ('[Write provenance contract] For '+self.write_tool+' declare evidence='+encode(dict(
            operation=self.demand['operation'], source_ids=[g[0] for g in self.demand['source_groups']],
            fields=self.demand['fields'], constant=self.demand['constant']))+'. Eligible original snapshots: '+encode(sources)+'. '
            'Read original numeric values. The declaration is not a computed answer; validation precedes writing.')

    def tools(self, tools):
        tools = copy.deepcopy(tools)
        found = False
        for t in tools:
            if t['function']['name'] != self.write_tool:
                continue
            found = True
            params = t['function']['parameters']
            params['properties']['evidence'] = dict(type='object', properties=dict(
                operation=dict(type='string', enum=['copy','sum','second_minus_first','add_constant']),
                source_ids=dict(type='array', items=dict(type='string'), minItems=1, maxItems=2),
                fields=dict(type='array', items=dict(type='string'), minItems=1, maxItems=2),
                constant=dict(type='integer')), required=['operation','source_ids','fields','constant'], additionalProperties=False)
            params.setdefault('required', []).append('evidence')
        if not found:
            raise ValueError('guarded write tool must be registered')
        return tools

    def validate(self, name, arguments):
        if name != self.write_tool:
            return arguments
        if not self.demand:
            raise ProvenanceRejected(self.error)
        proof = arguments.get('evidence')
        if not isinstance(proof, dict) or set(proof) != {'operation','source_ids','fields','constant'}:
            raise ProvenanceRejected('complete_evidence_declaration_required')
        demand = self.demand
        if proof['operation'] != demand['operation'] or proof['fields'] != demand['fields'] or type(proof['constant']) is not int or proof['constant'] != demand['constant']:
            raise ProvenanceRejected('operation_or_fields_do_not_match_task')
        ids = proof['source_ids']
        if not isinstance(ids, list) or len(ids) != len(demand['source_groups']) or any(not isinstance(s,str) or s not in g for s,g in zip(ids,demand['source_groups'])):
            raise ProvenanceRejected('source_version_does_not_match_task')
        vals = [self.facts[s]['body'][self.value_argument] for s in ids]
        operation = demand['operation']
        if operation == 'second_minus_first':
            expected = vals[1]-vals[0]
        elif operation == 'sum':
            expected = sum(self.facts[ids[0]]['body'][f] for f in demand['fields'])
        else:
            expected = vals[0]+demand['constant']
        if type(arguments.get(self.value_argument)) is not int or arguments[self.value_argument] != expected:
            raise ProvenanceRejected('write_value_does_not_match_declared_originals')
        return {k:v for k,v in arguments.items() if k != 'evidence'}
