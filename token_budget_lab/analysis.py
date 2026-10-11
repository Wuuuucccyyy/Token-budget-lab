"""Task-level descriptive statistics and paired bootstrap helpers."""
import random
import statistics
from collections import defaultdict

def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lower = int(index)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def interval(values):
    return [quantile(values, .025), quantile(values, .975)]


def latency_stats(rows, field='end_to_end_seconds'):
    values = [r[field] for r in rows if r.get(field) is not None]
    return dict(samples=len(values), mean_seconds=mean(values),
                p50_seconds=quantile(values, .5), p95_seconds=quantile(values, .95))


def cluster_ci(rows, field, seed=20261010, draws=2000):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[row['document_id']].append(row[field])
    means = [mean(v) for _, v in sorted(groups.items())]
    if not means:
        return None
    rng = random.Random(seed)
    samples = [mean(rng.choices(means, k=len(means))) for _ in range(draws)]
    return dict(documents=len(means), macro_mean=mean(means), ci95=interval(samples), draws=draws, seed=seed)

