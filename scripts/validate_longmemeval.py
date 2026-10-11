"""Validate an imported official subset; no model calls and no benchmark accuracy."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from token_budget_lab.datasets import digest
from token_budget_lab.experiment import code_hash
from token_budget_lab.memory import validate_messages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    bundle=json.loads(args.data.read_text(encoding='utf-8'))
    source=json.loads(args.source_manifest.read_text(encoding='utf-8'))
    if source['sha256']!=bundle['input_sha256'] or source['variant']!=bundle['variant']:
        raise ValueError('import/source mismatch')
    counts=Counter()
    message_counts=[]
    for s in bundle['sessions']:
        validate_messages(s['messages'])
        if not all(set(m)=={'role','content'} for m in s['messages']):
            raise ValueError('unexpected inference metadata/label')
        message_counts.append(len(s['messages']))
        for q in s['questions']:
            counts[q['kind']]+=1
            if q['evidence_positions'] is not None:
                if not all(0<=p<len(s['messages']) for p in q['evidence_positions']):
                    raise ValueError('invalid evidence position')
    result=dict(status='official_data_import_validation_only',has_model_accuracy=False,
        questions=len(bundle['sessions']),question_type_counts=dict(counts),source=source,
        original_question_ids=bundle['original_question_ids'],subset_sha256=digest(args.data),
        code_sha256=code_hash(),message_counts=message_counts,labels_removed_from_inference=True,
        oracle_warning=bundle['oracle_warning'],official_judge_run=False,
        note='Oracle contains evidence sessions already selected by the dataset; not a realistic retrieval comparison.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(questions=result['questions'],types=result['question_type_counts'],has_model_accuracy=False),indent=2))


if __name__=='__main__':main()
