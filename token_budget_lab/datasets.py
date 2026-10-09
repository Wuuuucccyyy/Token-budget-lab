"""Load synthetic requests or the document-normalized public-data bundle."""
import hashlib
import json
from pathlib import Path
from .benchmark import load_data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path, split=None):
    path = Path(path)
    if path.suffix == '.jsonl':
        if split:
            raise ValueError('synthetic JSONL has no document split')
        return load_data(path)
    bundle = json.loads(path.read_text(encoding='utf-8'))
    rows, seen = [], set()
    for q in bundle['questions']:
        if q['id'] in seen:
            raise ValueError('duplicate question ID')
        seen.add(q['id'])
        document = bundle['documents'][q['document_id']]
        if split and document['split'] != split:
            continue
        if not q['answers'] or any(not a or a not in document['context'] for a in q['answers']):
            raise ValueError('answer references must occur in source document')
        rows.append(dict(q, context=document['context'], split=document['split'], namespace='squad-article-v1'))
    if not rows:
        raise ValueError('empty dataset/split')
    return rows
