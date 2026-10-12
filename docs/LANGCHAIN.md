# LangChain 接入

v0.13.1 将上下文插件接到 LangChain 的 `create_agent`。`TokenBudgetMiddleware` 在模型调用前构造发送视图，在实际工具执行前后建立凭据并核验写入。LangChain 的规范消息始终保留完整工具结果，恢复来源来自实际执行结果的归档。

## 安装

Python 3.10+。核心包仍只依赖标准库，LangChain 为可选依赖：

```bash
python -m pip install ".[langchain]" -c configs/langchain_constraints.txt
python examples/langchain_agent.py
```

示例默认执行脚本化流程，验证图与工具接线，不请求模型，也不报告模型正确率或真实 token 收益。本机已设置 `DEEPSEEK_API_KEY` 时，使用 `--live` 执行最多十次模型调用的沙盒任务；`--without-plugin` 关闭上下文优化，保留相同执行与写入核验。

## 同一个 agent 切换插件

```python
from langchain.agents import create_agent
from token_budget_lab import DeepSeekChat
from token_budget_lab.langchain_model import DeepSeekChatModel
from token_budget_lab.langchain_adapter import (
    TokenBudgetContext, TokenBudgetMiddleware, business_tools,
)

tools = business_tools(tool_schemas)
with TokenBudgetMiddleware(tools, "local_results/langchain-archive") as middleware:
    agent = create_agent(
        DeepSeekChatModel(transport=DeepSeekChat(json_output=True)),
        tools=tools, middleware=[middleware], context_schema=TokenBudgetContext,
    )
    result = agent.invoke(
        {"messages": canonical_langchain_messages},
        context=TokenBudgetContext(
            tenant="user", session="conversation", enabled=True,
            handlers=tool_handlers, action_contract=contract, write_guard=guard,
            write_arguments={"write_report": ["total", "currency"]},
        ),
        config={"max_concurrency": 1, "recursion_limit": 70},
    )
    messages = result["messages"]
    metrics = result["token_budget_result"]
```

`tool_schemas`、`tool_handlers`、`contract` 和 `guard` 由宿主依据实际业务和当前用户要求构造，不能使用评测答案。完整可运行例子见 [langchain_agent.py](../examples/langchain_agent.py)，规则接口见 [INTEGRATION.md](INTEGRATION.md)。`enabled=False` 保留完整历史并关闭恢复工具定义。两次对照使用不同会话和独立业务状态，业务工具、模型、核验规则与步骤上限相同。

`business_tools` 把已有 Python handler 转成 LangChain 的 `StructuredTool`，同时保存 JSON 原始结果为工具 artifact。本轮数值来源规则适用于这种 JSON-result 接口。普通 LangChain 工具也可用于文本上下文外置与恢复；需要数值核验时，应通过上述接口提供实际结构化结果。

`write_arguments` 是宿主根据当前授权明确声明的字段范围。上例只允许 total 与 currency，即使工具 schema 提供可选 recipient，也不能擅自加入。违反范围的调用返回 write_arguments_rejected，handler 不执行；模型可在原步骤上限内修正，重试消耗保留。不传这个参数表示未启用字段范围检查，数值核验不会自动覆盖所有额外字段。

对于“唯一时写入并检查，缺失或重复时只读拒答”的任务，宿主可在执行合同中声明完整操作，再通过 `conditional_obligations=("write", "check")` 标记条件操作 ID。只有实际读取建立 missing/ambiguous 时，这些操作才记为 waived，不会伪造 completed 或执行凭据。unique 分支仍要求真正完成写入与检查；未完成的最终答案会被要求补做。该参数要求显式的 `FreshWriteGuard(..., conditional_read_only=True)`。

中间件也可以与其他 LangChain chat model 配合使用；真实实验使用已有 DeepSeek transport 的 `BaseChatModel` 包装，便于保留真实 usage、缓存命中与未命中分区、超时和无自动重试的行为。

## 运行与会话

- `agent.invoke` 与 `agent.ainvoke` 均提供适配。归档与插件操作在每个调用自己的单线程执行器中执行，避免跨线程使用 SQLite 连接。
- 同一任务内业务工具串行执行；独立租户与会话可以并发。当前评测设置 `max_concurrency=1`，不评估并发吞吐。
- 每次调用重新建立执行与写入凭据。相同会话继续对话时，应追加完整 `result["messages"]`，不能用压缩请求替换规范历史。
- 新任务重新建立条件授权；授权与补做提示仅进入模型请求，不写入规范历史。指定读取仍调用实际业务工具，历史恢复不能代替本轮读取。
- 中间件限制模型调用次数，默认十次；恢复最多三轮，并拦截完全相同的恢复请求。LangGraph 的 `recursion_limit` 统计图步骤，不能代替模型次数限制。
- 模型返回答案且必需操作尚未完成时，会继续要求补做；达到上限后仍未完成则返回失败状态。主机不会自动生成最终答案、改写错误数值或代执行写入。
- 首版支持文本消息、普通函数工具及 `FreshWriteGuard`。多模态消息、返回 LangGraph `Command` 的工具、`HistoricalWriteGuard` 的 schema 扩展与检查点中断后重启暂未验证。
- 使用 `with` 或在宿主关闭时调用 `middleware.close()`。中间件应放在其他修改消息的中间件之前；模型工具调用发生错误时不会自动付费重试。

## 对照实验

```bash
python -m token_budget_lab.langchain_experiment --plan
python -m token_budget_lab.langchain_experiment --output local_results/langchain-scoped-deepseek-v0131
```

计划使用十二个已用于调试的合成模板，各重复三次，插件关闭与开启共七十二个任务条件。只编译一个 LangChain agent，开关通过每次调用的 `TokenBudgetContext.enabled` 设置。模型、业务工具和必需操作、数值与条件授权核验保持一致，两组均最多十次模型调用，总请求上限为 720。

任务成功同时要求答案、文件状态、指定实际读取、操作顺序和最终回答完成。被宿主拦截的禁止写入尝试仍算违反用户约束。全部恢复、纠正和失败调用计入 token 与费用；耗时从 `agent.invoke` 前计到返回后，包含工具、检查、归档与审计日志。图编译耗时单独排除。

另外报告实际发生压缩的任务及相同任务的 Full 对照。三次重复不等于新增独立任务，结果不能替代开放业务任务的正确率验证。

扩展接口来自 [LangChain 自定义中间件文档](https://docs.langchain.com/oss/python/langchain/middleware/custom)。本轮依赖版本固定在 [langchain_constraints.txt](../configs/langchain_constraints.txt)，实验参数见 [langchain_deepseek.json](../configs/langchain_deepseek.json)。

当前验证状态与全部已发生请求的核算见 [LangChain 接入评测](../results/langchain/REPORT.md)。自带 agent 的 v0.12 结果保留在 [原运行器评测](../results/current/REPORT.md)，不能直接替代 LangChain 的开关对照。
