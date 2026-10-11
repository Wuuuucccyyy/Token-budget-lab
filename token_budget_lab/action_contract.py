"""Host-declared obligations backed by receipts from this invocation only."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from .memory import encode


def contains(actual, expected):
    """Typed recursive subset matching; True is not the integer 1."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and contains(actual[key], value) for key, value in expected.items())
    return type(actual) is type(expected) and actual == expected


@dataclass(frozen=True)
class RequiredAction:
    id: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    after: tuple[str, ...] = ()
    result_matches: Mapping[str, Any] = field(default_factory=dict)


class ActionRejected(ValueError):
    def __init__(self, missing):
        self.missing = missing
        super().__init__('required predecessor actions are incomplete')

    def result(self):
        return dict(error='action_prerequisite_missing', executed=False,
                    missing=self.missing,
                    note='Execute the named registered business tools now. Archive recovery is not execution.')


class ActionContract:
    """Explicit host policy, never a model-authored plan or a reference answer.

    Obligations are scoped to one run_agent call. `after` also makes an old
    check stale when its predecessor runs again. `before_tools` protects all
    invocations of listed tools, including calls with nonmatching arguments.
    """
    def __init__(self, actions, before_tools=None):
        self.actions = tuple(copy.deepcopy(actions))
        self.before_tools = copy.deepcopy(before_tools or {})
        self.by_id = {a.id: a for a in self.actions}
        if not self.actions or len(self.by_id) != len(self.actions):
            raise ValueError('nonempty, unique action ids required')
        for a in self.actions:
            if not a.id or not a.tool or a.tool.startswith(('context_', 'history_')):
                raise ValueError('obligations must name registered business tools')
            if not isinstance(a.arguments, dict) or not isinstance(a.result_matches, dict):
                raise ValueError('argument/result constraints must be objects')
            if len(set(a.after)) != len(a.after) or any(p not in self.by_id for p in a.after):
                raise ValueError('unknown or duplicate predecessor')
        for name, ids in self.before_tools.items():
            if not name or not ids or any(i not in self.by_id for i in ids):
                raise ValueError('unknown protected-tool predecessor')
        visiting, visited = set(), set()
        def visit(identity):
            if identity in visiting:
                raise ValueError('cyclic action contract')
            if identity in visited:
                return
            visiting.add(identity)
            for parent in self.by_id[identity].after:
                visit(parent)
            visiting.remove(identity)
            visited.add(identity)
        for identity in self.by_id:
            visit(identity)

    def start(self, registered, handlers):
        names = {a.tool for a in self.actions} | set(self.before_tools)
        if not names <= set(registered) or not names <= set(handlers):
            raise ValueError('contract tools require registered schemas and actual handlers')
        return ActionTracker(self)

    def definition(self):
        return dict(required=[dict(id=a.id, tool=a.tool, arguments=a.arguments,
                                   after=list(a.after), result_matches=a.result_matches)
                              for a in self.actions], before_tools=self.before_tools)

    def notice(self):
        return ('[Current task execution contract] '+encode(self.definition())+
                '\nThese obligations come from the current user request. Execute the business tools '
                'in this run with the specified arguments and order. Archived results, context_read, '
                'cached echoes and statements of completion are not execution receipts. '
                'For an explicitly required fresh read, call the business read first; recover only '
                'the needed fields of its returned source if the result is offloaded. '
                'A final answer is accepted only after all obligations succeed.')


class ActionTracker:
    def __init__(self, contract):
        self.contract = contract
        self.receipts = []
        self.last = {}

    def satisfied(self, identity):
        own = self.last.get(identity)
        return own is not None and all(
            self.satisfied(parent) and self.last[parent] < own
            for parent in self.contract.by_id[identity].after)

    def missing(self):
        return [a.id for a in self.contract.actions if not self.satisfied(a.id)]

    def validate(self, tool, arguments):
        predecessors = list(self.contract.before_tools.get(tool, ()))
        matching = [a for a in self.contract.actions if a.tool == tool and contains(arguments, a.arguments)]
        if matching and not any(all(self.satisfied(p) for p in a.after) for a in matching):
            for a in matching:
                predecessors.extend(a.after)
        missing = sorted({i for i in predecessors if not self.satisfied(i)})
        if missing:
            raise ActionRejected(missing)

    def needs_invocation(self, tool, arguments):
        return any(a.tool == tool and contains(arguments, a.arguments) and not self.satisfied(a.id)
                   for a in self.contract.actions)

    def record(self, tool, arguments, result=None, error_type=None):
        """Call only after an actual handler invocation, never on recovery/cache."""
        succeeded = error_type is None and not (
            isinstance(result, dict) and (result.get('error') or result.get('success') is False))
        sequence = len(self.receipts)
        matched = []
        if succeeded:
            for a in self.contract.actions:
                if (a.tool == tool and contains(arguments, a.arguments)
                        and (not a.result_matches or contains(result, a.result_matches))
                        and all(self.satisfied(p) for p in a.after)):
                    matched.append(a.id)
            # On a repeated tool/argument obligation, advance the successor
            # without refreshing its predecessor with the very same receipt.
            def ancestors(identity):
                parents = self.contract.by_id[identity].after
                return set(parents).union(*(ancestors(p) for p in parents))
            predecessors = set().union(*(ancestors(i) for i in matched))
            matched = [i for i in matched if i not in predecessors]
            for identity in matched:
                self.last[identity] = sequence
        receipt = dict(sequence=sequence, tool=tool, arguments=copy.deepcopy(arguments),
                       handler_returned=error_type is None, error_type=error_type,
                       succeeded=bool(succeeded), matched_actions=matched,
                       result_sha256=None if error_type else hashlib.sha256(encode(result).encode()).hexdigest())
        self.receipts.append(receipt)
        return receipt

    def state(self):
        missing = self.missing()
        return dict(complete=not missing, missing=missing,
                    completed=[a.id for a in self.contract.actions if a.id not in missing])

    def retry_notice(self):
        return ('[Host finalization rejected] Missing current-task execution receipts: '+
                encode(self.missing())+'. The preceding answer is provisional and the task is incomplete. '
                'Execute the required registered business tools with the arguments and order in the '
                'execution contract, then return a final answer. Do not recover old evidence instead '
                'of performing an explicitly required fresh read. The original step limit still applies.')
