"""Execute a LangChain file task; default to a scripted, zero-API wiring demo."""
import argparse
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from langchain.agents import create_agent
from token_budget_lab.action_tasks import execution_contract,update_current_files
from token_budget_lab.langchain_adapter import TokenBudgetContext,TokenBudgetMiddleware,business_tools
from token_budget_lab.langchain_model import DeepSeekChatModel,langchain_messages
from token_budget_lab.memory import encode
from token_budget_lab.memory_agent import DeepSeekChat
from token_budget_lab.tool_tasks import TOOLS,ToolSandbox
from token_budget_lab.transform_tasks import guard_from_query


class ScriptedDemo:
    """Demonstrate graph wiring only; this is not a model-quality measurement."""
    def __init__(self,rule):
        self.rule,self.step,self.value=rule,0,None
        self.last={}

    def __call__(self,messages,tools):
        if self.step==0:
            name,args=self.rule.read_tool,self.rule.read_arguments
        elif self.step==1:
            source=json.loads(next(m['content'] for m in reversed(messages) if m['role']=='tool'))
            if source.get('archived'):
                number=source['requested_evidence'][0]['original_value']
            else:
                selector=self.rule.selector
                records=[e for e in source['events'] if e[selector['match_key']]==selector['match_value']]
                if len(records)!=1:raise ValueError('The demonstration requires one exact event')
                number=records[0][selector['field']]
            self.value=format(Decimal(str(number)).quantize(Decimal('.01')),'f')
            name,args='write_report',dict(total=self.value,currency='USD')
        elif self.step==2:
            name,args='run_checks',{}
        else:
            return dict(message=dict(role='assistant',content=encode(dict(answer=self.value))),finish_reason='stop')
        identity=f'demo-{self.step}';self.step+=1
        return dict(message=dict(role='assistant',content='',tool_calls=[dict(id=identity,type='function',
            function=dict(name=name,arguments=encode(args)))]),finish_reason='tool_calls')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--without-plugin',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    episode=json.loads((root/'data/locator_tasks.json').read_text(encoding='utf-8'))['episodes'][0]
    with tempfile.TemporaryDirectory() as tmp:
        sandbox=ToolSandbox(Path(tmp)/'files',episode)
        history,_=sandbox.seed_history(episode)
        update_current_files(sandbox,episode['live_updates'])
        query=episode['checkpoints'][0]['query']
        history.append(dict(role='user',content=query))
        guard=guard_from_query(query)
        transport=DeepSeekChat(json_output=True) if args.live else ScriptedDemo(guard.rules[0])
        tools=business_tools(TOOLS)
        with TokenBudgetMiddleware(tools,Path(tmp)/'archive') as middleware:
            agent=create_agent(DeepSeekChatModel(transport=transport),tools=tools,
                middleware=[middleware],context_schema=TokenBudgetContext)
            result=agent.invoke(dict(messages=langchain_messages(history)),
                context=TokenBudgetContext('demo','session',enabled=not args.without_plugin,
                    handlers=sandbox.handlers(),action_contract=execution_contract(episode['request_actions']),write_guard=guard,
                    write_arguments={'write_report':['total','currency']}),
                config=dict(max_concurrency=1,recursion_limit=70))
            report=result['token_budget_result']
            print(json.dumps(dict(backend='deepseek' if args.live else 'scripted_wiring_only_no_API',
                status=report['status'],answer=report['answer'],plugin_enabled=report['enabled'],
                required_actions=report['action_contract']['completed'],model_dispatches=report['api_calls'],
                real_token_usage_available=args.live,actual_report=sandbox.read_report()['report']),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
