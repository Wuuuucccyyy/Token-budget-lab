"""Immutable canonical archive and message protocol helpers."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def scope_key(tenant, session):
    if not isinstance(tenant, str) or not tenant or not isinstance(session, str) or not session:
        raise ValueError('explicit tenant and session are required')
    return hashlib.sha256(encode([tenant, session]).encode()).hexdigest()


def validate_messages(messages):
    """Reject orphan tool results and unresolved calls before another model call."""
    pending = set()
    seen = set()
    for message in messages:
        role = message.get('role')
        if role not in {'system', 'developer', 'user', 'assistant', 'tool'}:
            raise ValueError('unsupported message role')
        if role == 'tool':
            call_id = message.get('tool_call_id')
            if call_id not in pending:
                raise ValueError('orphan tool result')
            pending.remove(call_id)
        elif pending:
            raise ValueError('unfinished tool-call group')
        if message.get('tool_calls'):
            if role != 'assistant':
                raise ValueError('only assistant messages may call tools')
            for call in message['tool_calls']:
                if not isinstance(call.get('id'), str) or call['id'] in seen:
                    raise ValueError('invalid/duplicate tool-call ID')
                if call.get('type') != 'function' or not isinstance(call.get('function'), dict):
                    raise ValueError('only function tool calls are supported')
                pending.add(call['id'])
                seen.add(call['id'])
    if pending:
        raise ValueError('resolve tool calls before preparing another model request')


def split_turns(messages):
    turns = []
    for position, message in enumerate(messages):
        if not turns or message['role'] == 'user':
            turns.append(dict(id=f'T{position}', positions=[], messages=[]))
        turns[-1]['positions'].append(position)
        turns[-1]['messages'].append(message)
    return turns


class HistoryArchive:
    """Local SQLite archive with append-only checks and session isolation."""
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS history (scope TEXT, position INTEGER, '
                                'value TEXT NOT NULL, sha TEXT NOT NULL, PRIMARY KEY(scope,position))')

    def close(self):
        self.connection.close()

    def ingest(self, scope, messages):
        existing = list(self.connection.execute('SELECT position,value,sha FROM history WHERE scope=? ORDER BY position', (scope,)))
        if len(messages) < len(existing):
            raise ValueError('canonical history shrank; do not replace it with compact messages')
        serialized = [encode(m) for m in messages]
        for position, value, sha in existing:
            if hashlib.sha256(value.encode()).hexdigest() != sha:
                raise ValueError('archive integrity check failed')
            if value != serialized[position]:
                raise ValueError('canonical history changed; use a new session')
        with self.connection:
            for position in range(len(existing), len(messages)):
                value = serialized[position]
                self.connection.execute('INSERT INTO history VALUES (?,?,?,?)',
                    (scope, position, value, hashlib.sha256(value.encode()).hexdigest()))

    def read(self, scope):
        messages = []
        for position, value, sha in self.connection.execute(
                'SELECT position,value,sha FROM history WHERE scope=? ORDER BY position', (scope,)):
            if position != len(messages) or hashlib.sha256(value.encode()).hexdigest() != sha:
                raise ValueError('archive integrity check failed')
            messages.append(json.loads(value))
        return messages


@dataclass
class PreparedHistory:
    messages: list
    mode: str
    reason: str
    baseline_units: int
    sent_units: int
    unit: str
    omitted_turn_ids: list
    restored_turn_ids: list
    preparation_seconds: float
    evidence_plan: dict = field(default_factory=dict)
    context_metadata: dict = field(default_factory=dict)


class CounterBackend:
    def __init__(self, encoding: str = "characters"):
        self.name = encoding
        self.encoder = None
        if encoding != "characters":
            import tiktoken  # Optional; missing dependency is an error, never a silent fallback.
            self.encoder = tiktoken.get_encoding(encoding)

    @property
    def unit(self) -> str:
        return "characters" if self.encoder is None else "tokens"

    def count(self, text: str) -> int:
        if self.encoder is None:
            return len(text)
        return len(self.encoder.encode(text, disallowed_special=()))

