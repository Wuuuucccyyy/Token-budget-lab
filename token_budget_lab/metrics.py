"""Reference-only scoring. These functions never select context."""
import re
import string
from collections import Counter


def normalize(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def score_answer(answer, references):
    """SQuAD-style English normalization; maximum over alternative references."""
    if not references:
        raise ValueError('at least one reference is required')
    prediction = normalize(answer)
    scores = []
    for reference in references:
        gold = normalize(reference)
        p, g = prediction.split(), gold.split()
        common = sum((Counter(p) & Counter(g)).values())
        f1 = 2 * common / (len(p) + len(g)) if p and g else float(p == g)
        scores.append((float(prediction == gold), f1))
    return dict(exact_match=max(x[0] for x in scores), f1=max(x[1] for x in scores))


def evaluate_answer(row, answer, selected):
    if 'answers' in row:
        return dict(**score_answer(answer, row['answers']),
                    answer_contains_gold=any(a.lower() in answer.lower() for a in row['answers']),
                    answer_span_retained=any(a.lower() in selected.lower() for a in row['answers']),
                    evidence_retained=None)
    return dict(exact_match=None, f1=None, answer_span_retained=None,
                answer_contains_gold=all(s in answer for s in row['expected']),
                evidence_retained=all(e in selected for e in row['evidence']))
