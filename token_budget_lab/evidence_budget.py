"""Indexed paragraph retrieval and a gold-free, heuristic evidence budget."""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .core import terms

STOPWORDS = set('a an the in of to for is was were did does do what which who when where how and or be on at by with that this it its are as'.split())


def retrieval_terms(text):
    return [t for t in terms(text) if t not in STOPWORDS]


def paragraph_spans(text):
    cursor = 0
    boundaries = [(m.start(), m.end()) for m in re.finditer(r'\n[ \t]*\n+', text)] + [(len(text), len(text))]
    for end, following in boundaries:
        start = cursor
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end-1].isspace():
            end -= 1
        if start < end:
            yield start, end
        cursor = following


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    start: int
    end: int
    text: str


def chunks_from_documents(documents, chunk_chars=1200, overlap_chars=160):
    if chunk_chars < 100 or not 0 <= overlap_chars < chunk_chars:
        raise ValueError('invalid chunk/overlap sizes')
    chunks = []
    for document_id, text in sorted(documents.items()):
        for start, end in paragraph_spans(text):
            cursor = start
            while cursor < end:
                stop = min(cursor+chunk_chars, end)
                if stop < end:
                    # Split on whitespace, not periods: retain initials/decimals.
                    boundary = max(text.rfind(' ', cursor+chunk_chars//2, stop),
                                   text.rfind('\n', cursor+chunk_chars//2, stop))
                    if boundary > cursor:
                        stop = boundary
                chunk_id = hashlib.sha256(f'{document_id}:{cursor}:{stop}'.encode()).hexdigest()[:16]
                chunks.append(Chunk(chunk_id, document_id, cursor, stop, text[cursor:stop]))
                if stop == end:
                    break
                cursor = max(cursor+1, stop-overlap_chars)
    return chunks


class EvidenceIndex:
    """BM25 index built once; inference sees the entire corpus, never gold IDs."""
    def __init__(self, documents, chunk_chars=1200, overlap_chars=160):
        self.chunks = chunks_from_documents(documents, chunk_chars, overlap_chars)
        self.postings = defaultdict(list)
        self.lengths = []
        for index, chunk in enumerate(self.chunks):
            counts = Counter(retrieval_terms(chunk.document_id.replace('_',' ')+' '+chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths)/max(1,len(self.lengths))

    def idf(self, term):
        return math.log(1+(len(self.chunks)-len(self.postings.get(term,[]))+.5)/(len(self.postings.get(term,[]))+.5))

    def search(self, query, top_k=24):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        scores = defaultdict(float)
        for term in set(retrieval_terms(query)):
            for index, frequency in self.postings.get(term,[]):
                denominator = frequency+1.5*(.25+.75*self.lengths[index]/(self.average_length or 1))
                scores[index] += self.idf(term)*frequency*2.5/denominator
        order = sorted(scores, key=lambda i:(-scores[i], self.chunks[i].id))[:top_k]
        return [dict(chunk=self.chunks[i], score=scores[i], rank=rank+1) for rank,i in enumerate(order)]

    def coverage(self, query, chunks):
        query_terms = set(retrieval_terms(query))
        available = {t for chunk in chunks for t in retrieval_terms(chunk.text)}
        total = sum(self.idf(t) for t in query_terms)
        return sum(self.idf(t) for t in query_terms & available)/total if total else 0.0


DEFAULT_POLICY = dict(small=1800, medium=3600, large=7200, weak_coverage=.6,
                      strong_coverage=.8, ambiguous_gap=.15, clear_gap=.35)


def validate_policy(policy):
    if any(not isinstance(policy[k], int) or isinstance(policy[k], bool) for k in ('small', 'medium', 'large')):
        raise ValueError('budget tiers must be integers')
    if not 0 < policy['small'] < policy['medium'] < policy['large']:
        raise ValueError('budget tiers must increase')
    if not 0 <= policy['weak_coverage'] <= policy['strong_coverage'] <= 1:
        raise ValueError('invalid coverage thresholds')
    if not 0 <= policy['ambiguous_gap'] <= policy['clear_gap'] <= 1:
        raise ValueError('invalid score-gap thresholds')


def choose_budget(query, candidates, index, policy=None):
    """Only query, retrieved chunks and index statistics are accepted."""
    policy = dict(DEFAULT_POLICY, **(policy or {}))
    validate_policy(policy)
    if not candidates:
        return dict(budget=policy['small'], tier='small', reason='no_candidates', top_coverage=0.0, score_gap=0.0)
    coverage = index.coverage(query,[candidates[0]['chunk']])
    top = candidates[0]['score']
    second = candidates[1]['score'] if len(candidates)>1 else 0
    gap = (top-second)/top if top else 0
    complex_query = bool(re.search(r'\b(compare|difference|between|both|why|how does|how did|except|before|after)\b|比较|区别|分别|同时|为什么|条件|除非|之前|之后',query.lower()))
    if complex_query:
        tier,reason='large','complex_question'
    elif coverage < policy['weak_coverage']:
        tier,reason='large','weak_term_coverage'
    elif gap < policy['ambiguous_gap']:
        tier,reason='large','ambiguous_candidates'
    elif coverage >= policy['strong_coverage'] and gap >= policy['clear_gap']:
        tier,reason='small','concentrated_lexical_match'
    else:
        tier,reason='medium','moderate_lexical_match'
    return dict(budget=policy[tier],tier=tier,reason=reason,top_coverage=coverage,score_gap=gap)


def render_evidence(chunks):
    return '\n\n'.join(f'[E{i}] {c.document_id} ({c.start}:{c.end})\n{c.text}' for i,c in enumerate(chunks,1))


def pack_evidence(candidates, budget):
    """Whole chunks with citation headers; budget is strictly in characters."""
    if budget < 0:
        raise ValueError('budget must be nonnegative')
    selected=[]
    for candidate in candidates:
        chunk=candidate['chunk']
        redundant = any(c.document_id == chunk.document_id and
                        max(0,min(c.end,chunk.end)-max(c.start,chunk.start))/min(c.end-c.start,chunk.end-chunk.start) >= .6
                        for c in selected)
        if redundant:
            continue
        if len(render_evidence(selected+[chunk])) <= budget:
            selected.append(chunk)
    return dict(context=render_evidence(selected),chunks=selected)


def source_paragraph_covered(row, documents, selected):
    """Evaluation-only: cover the gold source paragraph with union of intervals."""
    text=documents[row['document_id']]
    spans=[(start,end) for start,end in paragraph_spans(text)
           if hashlib.sha256(text[start:end].encode()).hexdigest()==row['source_paragraph_sha256']]
    if not spans:
        raise ValueError('source paragraph hash does not match corpus')
    intervals=sorted((c.start,c.end) for c in selected if c.document_id==row['document_id'])
    for start,end in spans:
        covered=start
        for left,right in intervals:
            if left > covered:
                break
            if right > covered:
                covered=right
            if covered>=end:
                return True
    return False
