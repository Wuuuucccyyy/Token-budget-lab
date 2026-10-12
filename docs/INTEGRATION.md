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

## 精准证据定位

v0.9 默认 `evidence_mode="structured"`。已知来源与事件标识时直接调用：

```python
evidence = plugin.invoke_context("user", "conversation", "context_find", {
    "source_id": source_id,
    "match_key": "event",
    "match_value": "scan_d-43",
    "fields": ["status_code"],
})
```

返回 matches、match_count、ambiguous 和 has_more。每个匹配包含实际 JSON pointer、字段路径及字段值。默认最多三个匹配，仍报告总数。重复 event 不自动选择；找不到是指定完整 JSON 来源中的精确匹配结果，不能代替全局信息不存在的判断。

匹配值目前为字符串。小于等于 1,000 字符的事件对象完整返回；更大的对象返回指定字段及路径，超过响应预算的字段使用 context_read 再恢复，不返回伪完整 JSON。

`context_search(query)` 对明确事件/id 返回结构化匹配，对自由文本返回命中中心片段及原文偏移。当前搜索参数仍只有 query、limit，不接受 source_id；已知 JSON 来源使用 context_find。单任务工具定义固定，来源 ID 必须来自当前会话的归档入口。

需要复查原恢复行为时显式设置 `evidence_mode="lexical"`，该对照不提供 context_find。

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

## 本轮写入值变换核验

```python
from token_budget_lab import FreshWriteGuard, WriteTransform

guard = FreshWriteGuard([WriteTransform(
    read_tool="read_log", read_arguments={"name": "scan_c"},
    write_tool="write_report", value_argument="total",
    selector={"kind": "unique_event", "array": "/events",
              "match_key": "event", "match_value": "scan_c-39",
              "field": "status_code"},
    operation="format_decimal", places=2,
    fixed_arguments={"currency": "USD"},
)])
# 在 run_agent 中增加 write_guard=guard，Full 对照也传同样的 guard。
```

规则由宿主根据当前用户授权的任务构造，只包含读取参数、字段选择与变换，不包含目标数值。调用实际读取 handler 后，宿主保存本轮来源 ID、原始结果指纹和完整返回；恢复历史、缓存回显和模型自报 evidence 不建立新来源。每次 run_agent 创建独立核验状态。

`unique_event` 在实际数组中精确匹配 event，必须恰好一个匹配，不根据编号猜下标。`pointer` 直接取指定 RFC6901 字段。`text_record` 在指定文本字段中查唯一 marker，再从到分号或换行之间的记录读取唯一 `field=value`；这是明确的文本格式约定，不是任意日志解析。

`format_decimal` 使用 Decimal 与 ROUND_HALF_EVEN，保留数值大小，例如整数 409 格式化为 `"409.00"`。`add_integer` 要求整数来源，按宿主声明的 constant 偏移。小数位数最多六位，数值大小最多 10¹⁸；不支持任意自然语言计算。

提交的值、类型、格式或固定参数不符时，返回 `write_value_rejected`，实际写 handler 不执行。可计算时反馈包含从真实原文重算的 expected_value，模型可在原步骤预算内修改调用。宿主不自动改写提交值或代执行写入；最终答案仍需单独评测。

`result["write_transform"]` 返回全部 accepted/rejected 事件，包括实际源指纹、字段位置、原值、重算值和提交值。读取报错后清除该读取的旧来源；会改变源文件的已知写工具在执行前使来源失效，覆盖部分修改后抛错的情况。源读取与写入之间发生的外部并发更新目前没有事务保证。

当前规则核验已配置的写工具、目标数值与 fixed_arguments，不自动覆盖其他工具或额外业务字段。任意工具的源修改关系需要宿主扩展；HistoricalWriteGuard 与 FreshWriteGuard 目前通过单一 write_guard 参数择一接入。

实验的 `guard_from_query` 仅识别冻结任务的几种明确措辞，没有通用自然语言解析能力。生产接入优先使用上面的显式结构规则。

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
