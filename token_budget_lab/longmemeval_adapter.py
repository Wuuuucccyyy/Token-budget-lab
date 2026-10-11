"""Import selected official memory questions, without passing answer labels to inference."""
import argparse
import json
import random
from pathlib import Path

from .datasets import digest

SOURCE='https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned'
KINDS=('knowledge-update','temporal-reasoning','abstention')


def convert(rows,limit=30,seed=20261011,variant='s',split='test'):
    if limit<1 or variant not in {'oracle','s','m'} or split not in {'dev','test'}:
        raise ValueError('invalid import settings')
    groups={kind:[] for kind in KINDS}
    seen=set()
    for row in rows:
        qid=str(row['question_id'])
        if qid in seen: raise ValueError('duplicate question_id')
        seen.add(qid)
        kind='abstention' if qid.endswith('_abs') else row['question_type']
        if kind in groups: groups[kind].append(row)
    rng=random.Random(seed)
    for group in groups.values(): rng.shuffle(group)
    selected=[]
    for i in range(max((len(g) for g in groups.values()),default=0)):
        for kind in KINDS:
            if i<len(groups[kind]): selected.append((kind,groups[kind][i]))
    selected=selected[:limit]
    if not selected: raise ValueError('no requested question types')
    sessions=[]
    for kind,row in selected:
        originals=row['haystack_sessions']; dates=row['haystack_dates']; ids=row['haystack_session_ids']
        if not len(originals)==len(dates)==len(ids): raise ValueError('inconsistent history metadata')
        messages=[dict(role='system',content='Answer from the dated history. Do not invent missing facts. Return only JSON {"answer":"short answer"}. If no answer is recorded, use UNKNOWN.')]
        evidence=[]
        for index in sorted(range(len(originals)),key=lambda i:(dates[i],str(ids[i]))):
            messages.append(dict(role='user',content=f'[Historical session: {ids[index]}; date: {dates[index]}]'))
            for raw in originals[index]:
                if raw.get('role') not in {'user','assistant'} or not isinstance(raw.get('content'),str):
                    raise ValueError('unsupported official message format')
                if raw.get('has_answer'): evidence.append(len(messages))
                # Copy only these two fields; has_answer and all other labels stay outside inference.
                messages.append(dict(role=raw['role'],content=raw['content']))
        qid=str(row['question_id'])
        reference=row['answer']
        if isinstance(reference,(dict,list)) or reference is None: raise ValueError('unsupported answer type')
        sessions.append(dict(id='lme-'+qid,split=split,messages=messages,questions=[dict(id=qid,kind=kind,
            query=f"Question date: {row['question_date']}\n{row['question']}",answers=['UNKNOWN' if kind=='abstention' else str(reference)],
            official_reference_answer=str(reference),evidence_positions=evidence or None)]))
    return dict(schema_version=1,source='official LongMemEval cleaned subset; adapter EM is not official judged accuracy',
        source_url=SOURCE,source_license='MIT',variant=variant,oracle_warning=variant=='oracle',
        selection_seed=seed,original_question_ids=[s['questions'][0]['id'] for s in sessions],
        split_note='local subset label, not an official train/test split',sessions=sessions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['oracle','s','m'],required=True)
    parser.add_argument('--limit',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20261011)
    args=parser.parse_args()
    rows=json.loads(args.input.read_text(encoding='utf-8'))
    if not isinstance(rows,list): raise ValueError('official JSON array required')
    bundle=convert(rows,args.limit,args.seed,args.variant)
    bundle['input_sha256']=digest(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(imported=len(bundle['sessions']),variant=args.variant,oracle=args.variant=='oracle',
        has_model_accuracy=False,input_sha256=bundle['input_sha256']),indent=2))


if __name__=='__main__': main()
