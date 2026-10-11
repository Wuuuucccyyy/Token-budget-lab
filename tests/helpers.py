"""Scripted transport fixtures, never model accuracy measurements."""
import copy
import json

class Script:
    last = {}
    def __init__(self, responses):
        self.responses, self.requests = responses, []

    def __call__(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        response = self.responses[len(self.requests)-1]
        message = dict(role='assistant', content=response if isinstance(response, str) else None)
        if not isinstance(response, str):
            message['tool_calls'] = [dict(id=f'{len(self.requests)}-{i}', type='function',
                function=dict(name=name, arguments=json.dumps(args))) for i, (name, args) in enumerate(response)]
        return dict(message=message, usage=dict(prompt_tokens=100, completion_tokens=10))


BUSINESS = dict(type='function', function=dict(name='finish', description='Complete the file task',
                parameters=dict(type='object', properties={}, additionalProperties=False)))
