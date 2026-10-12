"""Exact JSON object locations and query-centered original-text excerpts."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass

from .memory import encode
from .retrieval import EvidenceIndex, retrieval_terms


def escape_pointer(key):
    return str(key).replace('~', '~0').replace('/', '~1')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key')
        result[key] = value
    return result


@dataclass(frozen=True)
class ObjectLocation:
    source_id: str
    pointer: str
    value: dict


class EvidenceLocator:
    """Locations come from the actual tree, never event-number arithmetic.

    All sources remain eligible. This component does not select the effective
    version or bind a business write to a fresh read receipt.
    """
    identifier_keys = {'event', 'id', 'event_id', 'record_id', 'request_id'}

    def __init__(self, documents):
        self.documents = documents.copy()
        self.lexical = EvidenceIndex(documents)
        self.objects = []
        self.invalid_json_sources = set()
        self._identifiers = {}
        for sid, raw in sorted(documents.items()):
            try:
                value = json.loads(raw, object_pairs_hook=unique_object)
            except (ValueError, TypeError):
                self.invalid_json_sources.add(sid)
                continue
            stack = [('', value)]
            while stack:
                pointer, item = stack.pop()
                if isinstance(item, dict):
                    location = ObjectLocation(sid, pointer, item)
                    self.objects.append(location)
                    for key in self.identifier_keys:
                        identifier = item.get(key)
                        if isinstance(identifier, str) and identifier:
                            self._identifiers.setdefault(identifier, []).append(location)
                    stack.extend((pointer+'/'+escape_pointer(k), v) for k,v in reversed(list(item.items()))
                                 if isinstance(v, (dict,list)))
                elif isinstance(item, list):
                    stack.extend((pointer+'/'+str(i), item[i]) for i in range(len(item)-1,-1,-1)
                                 if isinstance(item[i], (dict,list)))

    def describe(self, location, fields=()):
        value, pointer = location.value, location.pointer
        record = encode(value)
        # Entire small objects are returned. Large objects have no clipped JSON
        # preview; requested small fields and their exact paths remain available.
        full = len(record) <= 1000
        selected, paths, omitted = {}, {}, []
        keys = fields or [k for k,v in value.items() if not isinstance(v,(dict,list))][:8]
        for key in keys:
            if key not in value:
                continue
            paths[key] = pointer+'/'+escape_pointer(key)
            candidate = {**selected, key:value[key]}
            if len(encode(candidate)) <= 800:
                selected[key] = copy.deepcopy(value[key])
            else:
                omitted.append(key)
        result = dict(source_id=location.source_id, pointer=pointer,
                      field_pointers=paths, fields=selected, omitted_fields=omitted,
                      record_complete=full, record_characters=len(record))
        if full:
            result['record'] = copy.deepcopy(value)
        return result

    def find(self, source_id, match_key, match_value, fields=(), limit=3):
        if source_id not in self.documents:
            raise ValueError('source not available for JSON lookup')
        if source_id in self.invalid_json_sources:
            raise ValueError('source is not unambiguous JSON; use original-text recovery')
        matches = [r for r in self.objects if r.source_id==source_id
                   and type(r.value.get(match_key)) is str and r.value[match_key]==match_value]
        rows = bounded_rows([self.describe(r, fields) for r in matches[:limit]])
        return dict(matches=rows,
                    match_count=len(matches), ambiguous=len(matches)>1,
                    has_more=len(matches)>len(rows),
                    note='Exact matches in this source only. Duplicate matches are not resolved automatically.')

    def search(self, query, limit=3):
        exact, seen = [], set()
        for identifier, locations in self._identifiers.items():
            if not re.search(r'(?<![A-Za-z0-9_.:-])'+re.escape(identifier)+r'(?![A-Za-z0-9_.:-])', query, re.I):
                continue
            for location in locations:
                identity = location.source_id, location.pointer
                if identity not in seen:
                    exact.append(location)
                    seen.add(identity)
        if exact:
            def fields(location):
                return [k for k in location.value if re.search(
                    r'(?<![A-Za-z0-9_])'+re.escape(k)+r'(?![A-Za-z0-9_])',query,re.I)][:8]
            rows = bounded_rows([dict(kind='json_object', **self.describe(r, fields(r))) for r in exact[:limit]])
            return dict(results=rows,
                        match_count=len(exact), ambiguous=len(exact)>1, has_more=len(exact)>len(rows),
                        note='Exact identifier matches; multiple sources can be different versions. Select evidence required by the current task.')
        results = []
        for ranked in self.lexical.search(query, limit):
            chunk = ranked['chunk']
            start, end = centered_span(chunk.text, query, self.lexical)
            results.append(dict(kind='text_excerpt', source_id=chunk.document_id,
                score=ranked['score'], text=chunk.text[start:end],
                offset=chunk.start+start, end_offset=chunk.start+end, partial=True))
        return dict(results=results, note='Query-centered partial original text; search misses do not establish absence.')


def centered_span(text, query, index, width=500):
    words = set(retrieval_terms(query))
    words.update(re.findall(r'[A-Za-z][A-Za-z0-9_.:-]*', query))
    candidates = []
    for word in words:
        match = re.search(re.escape(word), text, re.I)
        if match:
            candidates.append((index.idf(word), len(word), -match.start(), match.start()))
    anchor = max(candidates)[-1] if candidates else 0
    start = max(0, min(anchor-80, len(text)-width))
    return start, min(len(text), start+width)


def bounded_rows(rows, budget=3600):
    output = []
    for row in rows:
        if len(encode(output+[row])) > budget and 'record' in row:
            row = {k:v for k,v in row.items() if k!='record'}
            row.update(record_complete=False, record_omitted_for_budget=True)
        if len(encode(output+[row])) <= budget:
            output.append(row)
    return output
