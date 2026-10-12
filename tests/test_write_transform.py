import copy
import json
import tempfile
import unittest
from pathlib import Path

from helpers import Script
from token_budget_lab import ContextPlugin, FreshWriteGuard, HistoryArchive, WriteTransform, run_agent
from token_budget_lab.action_tasks import execution_contract
from token_budget_lab.action_experiment import plan
from token_budget_lab.context_plugin import source_id
from token_budget_lab.memory import encode
from token_budget_lab.tool_tasks import TOOLS
from token_budget_lab.transform_tasks import guard_from_query
from token_budget_lab.write_transform import WriteValueRejected


def rule(**changes):
    params=dict(read_tool='read_log',read_arguments=dict(name='scan_a'),write_tool='write_report',value_argument='total',
        selector=dict(kind='unique_event',array='/events',match_key='event',match_value='scan_a-15',field='status_code'),
        fixed_arguments=dict(currency='USD'))
    params.update(changes)
    return WriteTransform(**params)


def observed(guard,value,args=None,returned=True):
    args=dict(name='scan_a') if args is None else args
    guard.before_invocation('read_log',args)
    message=dict(role='tool',tool_call_id='actual-read',content=encode(value))
    guard.observe('read_log',args,value,message,3,returned)
    return source_id(3,message)


def value(code=409):
    return dict(events=[dict(event='noise',status_code=200),dict(event='scan_a-15',status_code=code)],padding='background '*200)


class WriteTransformTests(unittest.TestCase):
    def test_scale_errors_rejected_without_silent_repair(self):
        for raw,wrong,right in [(409,'4.09','409.00'),(204,'2.04','204.00')]:
            g=FreshWriteGuard([rule()]); sid=observed(g,value(raw))
            with self.assertRaises(WriteValueRejected) as caught:
                g.validate('write_report',dict(total=wrong,currency='USD'))
            error=caught.exception.result()
            self.assertEqual(error['expected_value'],right)
            self.assertEqual(error['source_location'],'/events/1/status_code')
            self.assertEqual(error['source_id'],sid)
            self.assertFalse(error['executed'])
            args=dict(total=right,currency='USD')
            self.assertEqual(g.validate('write_report',args),args)
            self.assertEqual((g.state()['accepted'],g.state()['rejected']),(1,1))

    def test_exact_event_not_array_position_and_immutable_source(self):
        g=FreshWriteGuard([rule()]); v=value();observed(g,v)
        v['events'][1]['status_code']=204
        g.validate('write_report',dict(total='409.00',currency='USD'))
        with self.assertRaises(WriteValueRejected):
            g.validate('write_report',dict(total='200.00',currency='USD'))

    def test_no_history_or_forged_proof_can_establish_freshness(self):
        g=FreshWriteGuard([rule()])
        with self.assertRaises(WriteValueRejected) as caught:
            g.validate('write_report',dict(total='409.00',currency='USD',evidence=dict(source_id='old')))
        self.assertEqual(caught.exception.detail['reason'],'successful_fresh_read_required')

    def test_wrong_file_failed_read_and_error_result_not_evidence(self):
        for args,returned,body in [(dict(name='scan_b'),True,value()),(None,False,value()),(None,True,dict(error='failed'))]:
            g=FreshWriteGuard([rule()]);observed(g,body,args,returned)
            with self.assertRaises(WriteValueRejected):g.validate('write_report',dict(total='409.00',currency='USD'))
        g=FreshWriteGuard([rule()]);observed(g,value());observed(g,value(),returned=False)
        with self.assertRaises(WriteValueRejected):g.validate('write_report',dict(total='409.00',currency='USD'))

    def test_latest_actual_read_supersedes_earlier_current_read(self):
        g=FreshWriteGuard([rule()]);observed(g,value(204));observed(g,value(409))
        with self.assertRaises(WriteValueRejected):g.validate('write_report',dict(total='204.00',currency='USD'))
        g.validate('write_report',dict(total='409.00',currency='USD'))

    def test_missing_duplicate_and_invalid_numeric_fields_reject(self):
        for body in [dict(events=[]),dict(events=value()['events']*2),value(True),value('NaN'),value('1e300'),value(float('inf')),value('UNKNOWN')]:
            g=FreshWriteGuard([rule()]);observed(g,body)
            with self.assertRaises(WriteValueRejected):g.validate('write_report',dict(total='409.00',currency='USD'))

    def test_format_and_fixed_currency_are_exact(self):
        g=FreshWriteGuard([rule()]);observed(g,value(409))
        for args in [dict(total='409',currency='USD'),dict(total=409,currency='USD'),dict(total='409.00',currency='CNY'),dict(recipient='a@b.invalid')]:
            with self.assertRaises(WriteValueRejected):g.validate('write_report',args)
        g.validate('write_report',dict(total='409.00',currency='USD'))

    def test_rounding_is_explicit_and_bounded(self):
        for raw,right in [('2.345','2.34'),('2.355','2.36'),('1000000000000000000','1000000000000000000.00')]:
            g=FreshWriteGuard([rule()]);observed(g,value(raw));g.validate('write_report',dict(total=right,currency='USD'))
        for changes in [dict(places=20),dict(operation='divide'),dict(constant=100),dict(selector=dict(kind='unknown'))]:
            with self.assertRaises(ValueError):rule(**changes)

    def test_integer_offset_and_before_partial_mutation_invalidation(self):
        r=WriteTransform('read_config',{},'write_config','timeout_ms',dict(kind='pointer',pointer='/config/timeout_ms'),operation='add_integer',constant=1)
        g=FreshWriteGuard([r]);msg=dict(role='tool',tool_call_id='r',content=encode(dict(config=dict(timeout_ms=1837))))
        g.observe('read_config',{},dict(config=dict(timeout_ms=1837)),msg,3,True)
        g.validate('write_config',dict(timeout_ms=1838))
        with self.assertRaises(WriteValueRejected):g.validate('write_config',dict(timeout_ms=1839))
        g.before_invocation('write_config',dict(timeout_ms=1838))
        with self.assertRaises(WriteValueRejected):g.validate('write_config',dict(timeout_ms=1838))

    def test_text_record_exact_marker_and_field(self):
        r=rule(selector=dict(kind='text_record',pointer='/log_text',marker='TAIL_MARKER',field='status_code'))
        g=FreshWriteGuard([r]);observed(g,dict(log_text='noise '*900+'TAIL_MARKER status_code=206; unrelated status_code=503'))
        g.validate('write_report',dict(total='206.00',currency='USD'))
        for text in ['TAIL_MARKER_A status_code=206;','TAIL_MARKER; status_code=206;','TAIL_MARKER status_code=206 status_code=503;',
                     'TAIL_MARKER status_code=206; TAIL_MARKER status_code=206;']:
            observed(g,dict(log_text=text))
            with self.assertRaises(WriteValueRejected):g.validate('write_report',dict(total='206.00',currency='USD'))

    def test_host_adapter_has_no_gold_or_branch_input(self):
        fixtures=json.loads(Path('data/locator_tasks.json').read_text(encoding='utf-8'))
        for e in fixtures['episodes']:
            cp=e['checkpoints'][0];g=guard_from_query(cp['query']);altered=copy.deepcopy(e)
            altered['checkpoints'][0].update(answer='FAKE',report=dict(total='999.00'),no_writes=True)
            self.assertEqual(g.notice(),guard_from_query(altered['checkpoints'][0]['query']).notice())
            self.assertNotIn('expected_value',g.notice())
        self.assertEqual(guard_from_query(fixtures['episodes'][9]['checkpoints'][0]['query']).rules[0].selector['kind'],'unique_event')
        with self.assertRaises(ValueError):guard_from_query('Unknown arithmetic task')

    def test_run_rejects_wrong_write_then_corrects_with_same_six_step_cap(self):
        rows=[dict(role='system',content='Use business tools'),dict(role='user',content='Read scan_a and format its exact event code.')]
        for compressed in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                archive=HistoryArchive(Path(tmp)/'a.db')
                plugin=ContextPlugin(archive,budget=2000,offload_threshold=512) if compressed else None
                writes=[];g=FreshWriteGuard([rule()])
                script=Script([[('read_log',dict(name='scan_a'))],[('write_report',dict(total='4.09',currency='USD'))],
                    [('write_report',dict(total='409.00',currency='USD'))],[('run_checks',{})],'{"answer":"409.00"}'])
                actions=execution_contract(dict(required=[dict(id='r',tool='read_log',arguments=dict(name='scan_a'))],before_tools=dict(write_report=['r'])))
                result=run_agent(rows,script,plugin,'t','s',mode='recover' if compressed else 'full',memory_tools=compressed,
                    max_steps=6,extra_tools=TOOLS,handlers=dict(read_log=lambda name:value(),write_report=lambda **a:writes.append(a) or dict(written=True),
                    run_checks=lambda:dict(passed=True)),action_contract=actions,write_guard=g)
                self.assertEqual(writes,[dict(total='409.00',currency='USD')])
                self.assertEqual(result['status'],'ok');self.assertEqual(result['api_calls'],5)
                self.assertEqual(result['prompt_tokens'],500)
                self.assertEqual(result['write_transform']['rejected'],1)
                self.assertEqual(len(result['action_contract']['receipts']),3)
                self.assertEqual(g.sources,{})  # Controller state is run-local.
                archive.close()

    def test_recovery_echo_never_calls_observer_or_write_handler(self):
        rows=[dict(role='system',content='Use tools'),dict(role='user',content='Read scan_a'),
            dict(role='assistant',content=None,tool_calls=[dict(id='old',type='function',function=dict(name='read_log',arguments=encode(dict(name='scan_a'))))]),
            dict(role='tool',tool_call_id='old',content=encode(value())),dict(role='assistant',content='Recorded'),dict(role='user',content='Write current value')]
        sid=source_id(3,rows[3]);writes=[]
        with tempfile.TemporaryDirectory() as tmp:
            archive=HistoryArchive(Path(tmp)/'a.db');p=ContextPlugin(archive,budget=2000,offload_threshold=512)
            script=Script([[('context_find',dict(source_id=sid,match_key='event',match_value='scan_a-15',fields=['status_code']))],
                [('write_report',dict(total='409.00',currency='USD'))],'{"answer":"409.00"}'])
            result=run_agent(rows,script,p,'t','s',max_steps=3,extra_tools=TOOLS,
                handlers=dict(read_log=lambda name:value(),write_report=lambda **a:writes.append(a)),write_guard=FreshWriteGuard([rule()]))
            self.assertEqual(writes,[])
            self.assertEqual(result['write_transform']['events'][0]['reason'],'successful_fresh_read_required')
            archive.close()

    def test_preflight_missing_business_handler_rejects_before_model(self):
        g=FreshWriteGuard([rule()])
        with self.assertRaises(ValueError):g.start({t['function']['name']:t['function'] for t in TOOLS},{})

    def test_experiment_uses_same_rules_for_full_and_compressed_guard(self):
        config=json.loads(Path('configs/transform_deepseek.json').read_text(encoding='utf-8'))
        episodes,manifest=plan(config)
        self.assertEqual(len(episodes),12)
        self.assertEqual(manifest['config']['write_transform_arms'],['full','context_transform'])
        self.assertEqual(manifest['config']['max_steps'],6)
        config['write_transform_arms']=['context_transform']
        with self.assertRaises(ValueError):plan(config)


if __name__=='__main__':unittest.main()
