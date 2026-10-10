import unittest
from token_budget_lab.analysis import latency_stats, quality_screen, summarize


class AnalysisTests(unittest.TestCase):
    def test_latency_quantiles_keep_cache_hits_separate(self):
        rows=[dict(end_to_end_seconds=1),dict(end_to_end_seconds=2),dict(end_to_end_seconds=9)]
        result=latency_stats(rows)
        self.assertEqual(result['p50_seconds'],2)
        self.assertAlmostEqual(result['p95_seconds'],8.3)
        self.assertIsNone(latency_stats([])['p95_seconds'])

    def test_equal_point_estimate_is_not_quality_equivalence(self):
        interval=dict(documents=10,ci95=[-.04,.04])
        self.assertEqual(quality_screen('deepseek','exact_match',interval,100,100,0),'inconclusive')
        self.assertEqual(quality_screen('deepseek','exact_match',interval,100,100,0,allowed_drop=.05),'meets_configured_screen')

    def test_quality_loss_and_incomplete_runs_cannot_pass(self):
        interval=dict(documents=10,ci95=[-.2,-.08])
        self.assertEqual(quality_screen('deepseek','exact_match',interval,100,100,0),'below_configured_screen')
        self.assertEqual(quality_screen('deepseek','exact_match',interval,100,80,0),'incomplete')
        self.assertEqual(quality_screen('deepseek','exact_match',interval,100,100,1),'incomplete')

    def test_proxy_and_zero_baseline_do_not_approve_task_quality(self):
        interval=dict(documents=10,ci95=[0,0])
        self.assertEqual(quality_screen('offline','exact_match',interval,100,100,0),'proxy_only')
        self.assertEqual(quality_screen('deepseek','answer_contains_gold',interval,100,100,0),'proxy_only')
        self.assertEqual(quality_screen('deepseek','exact_match',interval,100,100,0,baseline_score=0),'uninformative_baseline')

    def test_summary_reports_quality_before_efficiency_with_hit_latencies(self):
        manifest=dict(config=dict(backend='offline',strategies=[dict(name='full',method='full')]),question_ids=['a','b'])
        rows=[]
        for i,hit,elapsed in [('a',False,2),('b',True,.01)]:
            rows.append(dict(event='result',strategy='full',id=i,document_id=i,status='ok',cache_hit=hit,
                             input_units=0 if hit else 100,baseline_input_units=100,original_context_units=90,
                             selected_context_units=90,compression_seconds=0,answer_contains_gold=True,
                             exact_match=0,f1=.2,end_to_end_seconds=elapsed))
        run=summarize(rows,manifest)['runs'][0]
        self.assertEqual(run['latency_cache_hit']['p50_seconds'],.01)
        self.assertEqual(run['latency_cache_miss']['p50_seconds'],2)
        self.assertEqual(run['quality_screen'],'proxy_only')
