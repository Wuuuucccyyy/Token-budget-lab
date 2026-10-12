"""Frozen current evaluation fixtures and host-declared obligations."""
import copy
import json
from .experiment import ROOT
from .action_contract import ActionContract, RequiredAction

def build_action_tasks():
    return json.loads((ROOT/"data/action_contract_tasks.json").read_text(encoding="utf-8"))

def execution_contract(request_actions):
    return ActionContract([RequiredAction(**a) for a in request_actions['required']], request_actions.get('before_tools'))


def update_current_files(sandbox, updates):
    for update in updates:
        if update['file'] not in {'config.json', 'scan_a.json', 'scan_b.json', 'scan_c.json', 'scan_d.json'}:
            raise ValueError('unknown fixture file')
        value = sandbox._read(update['file'])
        if 'replace' in update:
            value = copy.deepcopy(update['replace'])
        elif 'reorder' in update:
            order = update['reorder']
            if sorted(order) != list(range(len(value['events']))):
                raise ValueError('reorder must be a complete array permutation')
            value['events'] = [value['events'][i] for i in order]
        else:
            target = value['events'][update['event']] if 'event' in update else value
            target.update(copy.deepcopy(update['fields']))
        sandbox._write(update['file'], value)

