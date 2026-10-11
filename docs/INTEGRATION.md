# Python 宿主接入

核心接口从 `token_budget_lab` 导入。宿主负责用户身份、完整消息、模型 transport、工具 schema 与实际 handler；插件只处理发送视图、证据恢复和明确声明的检查。

## 模型与工具

`transport(messages, tools)` 返回 `message`、`usage`、`finish_reason` 等字段。message 使用标准 assistant/tool_calls 结构。内置 `DeepSeekChat` 从本机 `DEEPSEEK_API_KEY` 读取凭证，不写入仓库；发生错误不自动重发。

`extra_tools` 是函数工具定义，`handlers` 以工具名映射到 Python 可调用对象。所有需要约束的业务调用都应经过 `run_agent`；旁路调用无法产生受检查的凭据。

```python
from token_budget_lab import ContextPlugin, HistoryArchive, DeepSeekChat, run_agent

archive = HistoryArchive("local_results/history.sqlite3")
plugin = ContextPlugin(archive, budget=9000, offload_threshold=1800,
                       offload_tools=["read_log"])
try:
    result = run_agent(
        messages, DeepSeekChat(json_output=True), plugin,
        tenant="user", session="conversation", max_steps=6,
        max_recovery_rounds=3, extra_tools=schemas, handlers=handlers,
    )
    messages = result["messages"]
finally:
    archive.close()
```

相同用户会话应追加规范历史，不能用插件输出的压缩视图覆盖 `messages`。单独调用 `before_model` 时，宿主还需注册 `plugin.context_tools`、分发 `invoke_context` 并追加完整调用链。

## 本轮必需操作

```python
from token_budget_lab import ActionContract, RequiredAction

contract = ActionContract([
    RequiredAction("scan", "read_log", {"name": "scan_c"}),
    RequiredAction("report", "write_report", after=("scan",),
                   result_matches={"written": True}),
    RequiredAction("checks", "run_checks", after=("report",),
                   result_matches={"passed": True}),
], before_tools={"write_report": ["scan"], "run_checks": ["report"]})

# 在上面的 run_agent 中增加 action_contract=contract。
```

合同必须来自当前用户授权的操作。它不读取参考答案，不采纳模型自拟计划；目前由宿主明确声明，尚未自动理解任意自由文本。

参数和结果按带类型的子集匹配，`True` 不等于整数 `1`。实际调用异常、返回非空 error 或 success=false 不计完成；通过 result_matches 指定业务成功条件。异步提交完成不等于异步任务完成。

`before_tools` 保护指定工具的所有调用；`after` 要求当前操作在依赖之后执行。同工具相同参数的两次必需读取需要顺序依赖，分别产生凭据。每次 `run_agent` 都重新建立本轮状态。

`result["action_contract"]` 提供 completed、missing、complete 和 receipts。status 为 ok 才接受最终答案；缺少操作且步骤耗尽时为 action_incomplete。所有补做请求计入 api_calls 和实际 usage。

## 历史配置核验

```python
from token_budget_lab import HistoricalWriteGuard

guard = HistoricalWriteGuard(messages)
# 在 run_agent 中增加 write_guard=guard。
```

默认核验 `read_config` → `write_config(timeout_ms)`。模型须声明 evidence，包括来源 ID、字段、操作及常量。支持明确的最近/首次快照、截至日期、两日期差值、两个整数域相加及整数偏移。

核验使用当前任务开始前的原始读取，不用本轮重新读写替代指定历史版本。未知运算、主体不明或版本冲突会拒绝写入。它不核验任意自然语言答案，也不自动覆盖其他业务 schema。

## 完整历史对照

```python
result = run_agent(
    messages, transport, None, tenant="user", session="conversation",
    mode="full", memory_tools=False, max_steps=6,
    extra_tools=schemas, handlers=handlers, action_contract=contract,
)
```

两组采用同样的业务工具、执行合同和任务成功判定。`snapshot_read_tools` 仅用于宿主确认在任务内稳定的无参数读工具；任何其他业务操作会先使缓存快照失效。明确要求再次实际读取时仍执行 handler。

可运行示例见 [python_agent.py](../examples/python_agent.py)。
