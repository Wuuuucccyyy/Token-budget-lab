import copy
import json
import tempfile
import unittest
from pathlib import Path

from token_budget_lab import HistoryArchive
from token_budget_lab.memory import encode, scope_key, validate_messages


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = HistoryArchive(Path(self.tmp.name)/'history.db')
        self.scope = scope_key('tenant', 'session')
        self.rows = [dict(role='system', content='Preserve constraints.'), dict(role='user', content='Read file.')]

    def tearDown(self):
        self.archive.close()
        self.tmp.cleanup()

    def test_append_roundtrip_preserves_original_messages(self):
        self.archive.ingest(self.scope, self.rows)
        self.rows.append(dict(role='assistant', content='Recorded.'))
        self.archive.ingest(self.scope, self.rows)
        self.assertEqual(self.archive.read(self.scope), self.rows)

    def test_changed_or_shrunk_canonical_history_is_rejected(self):
        self.archive.ingest(self.scope, self.rows)
        changed = copy.deepcopy(self.rows)
        changed[-1]['content'] = 'Replaced by summary'
        for bad in (changed, self.rows[:1]):
            with self.assertRaises(ValueError):
                self.archive.ingest(self.scope, bad)

    def test_storage_corruption_is_rejected_on_read_and_append(self):
        self.archive.ingest(self.scope, self.rows)
        self.archive.connection.execute('UPDATE history SET value=? WHERE scope=? AND position=0', (encode({'role':'system','content':'corrupt'}), self.scope))
        with self.assertRaises(ValueError):
            self.archive.read(self.scope)
        with self.assertRaises(ValueError):
            self.archive.ingest(self.scope, self.rows)

    def test_sessions_and_tenants_do_not_share_archived_sources(self):
        self.archive.ingest(self.scope, self.rows)
        self.assertEqual(self.archive.read(scope_key('other', 'session')), [])
        self.assertEqual(self.archive.read(scope_key('tenant', 'other')), [])
        for tenant, session in (('', 'a'), ('a', ''), (None, 'a')):
            with self.assertRaises(ValueError):
                scope_key(tenant, session)

    def test_tool_call_groups_must_be_complete_and_ids_unique(self):
        call = dict(role='assistant', content=None, tool_calls=[dict(id='c', type='function', function=dict(name='read', arguments='{}'))])
        result = dict(role='tool', tool_call_id='c', content='{}')
        validate_messages(self.rows+[call, result])
        for bad in (self.rows+[call], self.rows+[result], self.rows+[call, self.rows[-1]], self.rows+[call, result, call, result]):
            with self.assertRaises(ValueError):
                validate_messages(bad)
