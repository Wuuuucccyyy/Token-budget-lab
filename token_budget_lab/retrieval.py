"""Bounded lexical retrieval over original archived tool sources."""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

def terms(text: str) -> list[str]:
    """English words plus Chinese characters/bigrams; not an LLM tokenizer."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run)
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words

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

