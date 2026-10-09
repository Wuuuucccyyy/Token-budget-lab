import hashlib
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Rebuild the article-level SQuAD adaptation from an official v1.1 dev JSON file.')
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'squad_article.json')
args = parser.parse_args()
source = json.loads(args.source.read_text(encoding='utf-8'))
source_hash = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
if source_hash != 'f010a6aa7da00e11991ab7b779ccee74d433c68b8f653777724217d0ab18dfad':
    raise ValueError('Source differs from the pinned SQuAD v1.1 dev content')
rng = random.Random(20261010)
articles = sorted(source['data'], key=lambda a: a['title'])
rng.shuffle(articles)
documents, questions = {}, []
for i, article in enumerate(articles):
    doc_id = article['title']
    split = 'train' if i < 28 else 'dev' if i < 38 else 'test'
    context = '\n\n'.join(p['context'] for p in article['paragraphs'])
    documents[doc_id] = dict(context=context, split=split)
    candidates = [(q, p['context']) for p in article['paragraphs'] for q in p['qas']]
    selected = rng.sample(sorted(candidates, key=lambda pair: pair[0]['id']), min(10, len(candidates)))
    for q, paragraph in selected:
        questions.append(dict(id=q['id'], document_id=doc_id, query=q['question'],
                              answers=list(dict.fromkeys(a['text'] for a in q['answers'])),
                              source_paragraph_sha256=hashlib.sha256(paragraph.encode()).hexdigest()))
bundle = dict(schema_version=1, source='SQuAD v1.1 dev',
              source_url='https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json',
              source_git_blob='e9a3f913ad1468ebe105b891334ca7b0bc0e2510',
              license='CC-BY-SA-4.0', seed=20261010,
              transformation='Join every paragraph in each article; sample 10 questions per article without answer-based filtering; split by article.',
              documents=documents, questions=questions)
target = args.output
target.write_bytes((json.dumps(bundle, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(len(documents), len(questions), target.stat().st_size)
