import unittest
from token_budget_lab.cache_cost import cache_cost


class CacheCostTests(unittest.TestCase):
    rates=dict(hit=.003,miss=.15,output=.6)

    def test_cache_hit_stays_in_token_count_but_has_separate_price(self):
        row=dict(prompt_tokens=1000,prompt_cache_hit_tokens=900,prompt_cache_miss_tokens=100,completion_tokens=20)
        result=cache_cost([row],self.rates)
        self.assertAlmostEqual(result['estimated_usd'],.0000297)
        self.assertEqual(result['cache_hit_fraction'],.9)

    def test_missing_or_inconsistent_partitions_never_receive_zero_price(self):
        for row in ({},dict(prompt_tokens=100,prompt_cache_hit_tokens=90,prompt_cache_miss_tokens=20,completion_tokens=2)):
            result=cache_cost([row],self.rates)
            self.assertIsNone(result['estimated_usd'])
            self.assertEqual(result['cache_partition_invalid_or_missing_requests'],1)
