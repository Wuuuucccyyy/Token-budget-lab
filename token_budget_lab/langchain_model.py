"""Text-message conversion and a metered DeepSeek model for LangChain."""
from __future__ import annotations

import copy
import json
import threading
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from .memory import encode


def canonical_messages(messages):
    """Keep tool IDs and exact text; reject unsupported content instead of dropping it."""
    result = []
    for message in messages:
        if not isinstance(message.content, str):
            raise ValueError('The first LangChain adapter supports text messages only')
        if isinstance(message, AIMessage):
            if message.invalid_tool_calls:
                raise ValueError('Invalid model tool arguments cannot be archived')
            value = dict(role='assistant', content=message.content)
            if message.tool_calls:
                value['tool_calls'] = [dict(id=call['id'], type='function', function=dict(
                    name=call['name'], arguments=encode(call['args']))) for call in message.tool_calls]
        elif isinstance(message, ToolMessage):
            value = dict(role='tool', content=message.content, tool_call_id=message.tool_call_id)
        elif isinstance(message, HumanMessage):
            value = dict(role='user', content=message.content)
        elif isinstance(message, SystemMessage):
            value = dict(role='system', content=message.content)
        elif isinstance(message, ChatMessage) and message.role == 'developer':
            value = dict(role='developer', content=message.content)
        else:
            raise ValueError('Unsupported LangChain message role')
        result.append(value)
    return result


def langchain_messages(messages):
    result = []
    for message in messages:
        role, content = message['role'], message.get('content') or ''
        if role == 'assistant':
            calls = [dict(id=c['id'], name=c['function']['name'],
                          args=json.loads(c['function']['arguments']), type='tool_call')
                     for c in message.get('tool_calls', [])]
            result.append(AIMessage(content=content, tool_calls=calls))
        elif role == 'tool':
            result.append(ToolMessage(content=content, tool_call_id=message['tool_call_id']))
        elif role == 'user':
            result.append(HumanMessage(content=content))
        elif role == 'system':
            result.append(SystemMessage(content=content))
        elif role == 'developer':
            result.append(ChatMessage(role=role, content=content))
        else:
            raise ValueError('Unsupported canonical message role')
    return result


class DeepSeekChatModel(BaseChatModel):
    """Use the existing transport inside a genuine LangChain create_agent graph.

    No automatic retries or streaming. BaseChatModel's async fallback runs this
    blocking transport in an executor. The lock protects its per-call metadata.
    """
    transport: Any = Field(exclude=True, repr=False)
    model_name: str = 'deepseek-flash'
    _transport_lock: Any = PrivateAttr(default_factory=threading.RLock)

    @property
    def _llm_type(self):
        return 'token-budget-deepseek'

    @property
    def _identifying_params(self):
        return dict(model=self.model_name)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice not in (None, 'auto') or kwargs:
            raise ValueError('This transport supports automatic tool choice only')
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError('Stop sequences are not supported by this transport')
        with self._transport_lock:
            response = self.transport(canonical_messages(messages), kwargs.get('tools', []))
        raw = response['message']
        message = langchain_messages([raw])[0]
        if not isinstance(message, AIMessage):
            raise ValueError('Model transport must return an assistant message')
        usage = response.get('usage')
        if usage is not None:
            message.usage_metadata = dict(input_tokens=usage['prompt_tokens'],
                output_tokens=usage['completion_tokens'],
                total_tokens=usage['prompt_tokens'] + usage['completion_tokens'])
        message.response_metadata = dict(finish_reason=response.get('finish_reason'),
            response_model=response.get('response_model', self.model_name),
            token_budget_usage=copy.deepcopy(usage),
            api_latency_seconds=response.get('api_latency_seconds'))
        return ChatResult(generations=[ChatGeneration(message=message)])
