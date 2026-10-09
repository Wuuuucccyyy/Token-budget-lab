"""Export selected public-data retrieval failures and a blank review sheet."""
import argparse
import csv
import json
from pathlib import Path
from .datasets import load_rows
from .experiment import ROOT


def export(run, output):
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    config=manifest['config']
    rows={r['id']:r for r in load_rows(ROOT/config['data'],config.get('split'))}
    records=[json.loads(s) for s in (run/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[r for r in records if r['event']=='result' and r['status']!='error']
    output.mkdir(parents=True,exist_ok=True)
    fields=['id','strategy','question','references','answer','factual_correct','evidence_sufficient','notes']
    with (output/'manual_review.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for record in records:
            row=rows[record['id']]
            writer.writerow(dict(id=row['id'],strategy=record['strategy'],question=row['query'],
                                 references=' | '.join(row.get('answers',row.get('expected',[]))),answer=record['answer']))
    selected=[]
    seen=set()
    for record in records:
        if record['strategy'] not in {'bm25-0.5','neighbor1-0.5'} or record['answer_contains_gold'] or record['id'] in seen:
            continue
        selected.append(record)
        seen.add(record['id'])
        if len(selected)==6:
            break
    lines=['# 公开数据失败案例','',
           '以下案例按运行日志顺序选择，限定 50% BM25 与邻域方法中答案片段未命中的前六个不同问题；不是随机代表性抽样。回答来自离线句子检索器。',
           '人工评分表预留字段为空，尚未开展人工正确性评审。公开数据及其答案摘录按 CC BY-SA 4.0 使用，来源见 data/PUBLIC_DATA.md。','']
    for r in selected:
        row=rows[r['id']]
        lines += [f"## {r['strategy']} · {r['id']}",'',f"文章：{row['document_id']}",'',f"问题：{row['query']}",'',
                  '可接受答案：'+' / '.join(row['answers']),'','检索器返回：','', '> '+r['answer'].replace('\n','\n> '),'',
                  f"答案原文仍在压缩上下文中：{'是' if r['answer_span_retained'] else '否'}；实际上下文比例：{r['selected_context_units']/r['original_context_units']:.1%}。",'',
                  '诊断：'+('答案字符串尚在上下文中，失败发生在离线回答选择环节；不能直接归因于压缩丢失答案。' if r['answer_span_retained'] else '压缩后已找不到参考答案字符串；需要检查相关句排序、预算竞争和分句边界。'),'']
    (output/'FAILURE_CASES.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.run,args.output)


if __name__=='__main__':
    main()
