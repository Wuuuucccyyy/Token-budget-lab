"""Audit provider cache partitions before repricing reported API usage."""


def cache_cost(usages, rates):
    hit = miss = output = 0
    invalid = 0
    for u in usages:
        values = [u.get(k) for k in ('prompt_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens')]
        if any(type(v) is not int or v < 0 for v in values) or values[1]+values[2] != values[0]:
            invalid += 1
            continue
        hit += values[1];miss += values[2];output += values[3]
    return dict(cache_hit_tokens=hit,cache_miss_tokens=miss,completion_tokens_with_cache_partition=output,
        cache_partition_invalid_or_missing_requests=invalid,
        cache_hit_fraction=None if invalid or not hit+miss else hit/(hit+miss),
        estimated_usd=None if invalid else (hit*rates['hit']+miss*rates['miss']+output*rates['output'])/1_000_000,
        pricing='fixed-rate repricing of reported usage; not an invoice')
