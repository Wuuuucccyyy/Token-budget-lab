# 工具结果外置、版本核验与缓存稳定性

v0.6 增加 `ContextPlugin`，作为原有 `MemoryPlugin` 之外的显式实验接口。核心安装只依赖 Python 标准库。任务正确性优先；效率同时检查实际 API token、缓存分项费用估算和端到端耗时。

## 工具结果进入上下文前处理

宿主始终追加完整的原始 assistant/tool 消息。调用 `before_model` 时，插件先把宿主声明的较大工具结果转换成短归档入口，再计算历史视图。原始结果保留在按租户、会话隔离的 SQLite 中，工具调用 ID 和返回消息角色不变。

默认仅外置超过 1,800 字符的 `read_log` 结果。配置、订单、报告和检查结果仍直接可见；其他工具须由宿主明确加入 `offload_tools`。写工具原文不默认外置，也不会为了恢复信息重新执行历史工具。

`context_read(source_id, pointer, offset, limit)` 直接读取原文或 RFC6901 JSON pointer，单次返回最多 2,000 字符。返回来源、偏移、是否还有后续页。例如 `/events/37/status_code` 只读该字段，避免先搜索整轮 ID，再把整个日志恢复进上下文。

`context_search(query, limit)` 使用原始业务工具结果建立词面索引，返回最多三个相关片段和来源 ID。检索结果本身不再次进入索引；搜索未命中不能证明信息不存在。模型可能需要换查询、分页，或明确表示证据不足。

## 保持一个任务内的请求稳定

每个新 user 任务开始时，先对已经外置大结果的历史判断是否需要进一步筛选。已支持的主体、字段及时间需求可选择来源整轮；不支持时保留小型原始证据和指令，大日志仍外置。

同一任务开始后，旧前缀和工具定义保持固定，只追加新消息；新产生的大结果采用相同的稳定归档入口。重复稳定读取返回已有结果，不通过反复移除工具定义或把动态状态通知插入 system 前缀来控制循环。其他业务操作在调用前清空稳定读取快照，包括可能部分修改后报错的操作。

9,000 字符是软阈值，不是最终模型输入 token 上限；系统指令、工具定义、用户要求及必要证据还会占用输入。保留关键内容后超出阈值时单独记录。新任务可能重新选择历史，因此这里不保证整段会话永远具有相同前缀，也不保证服务商一定命中缓存。

## 历史版本写入核验

`HistoricalWriteGuard` 是独立的业务核验接口，宿主显式指定读取工具、写工具、原文结构和数值字段。参考实现覆盖 `read_config` → `write_config(timeout_ms)`。

模型写入时必须声明 `evidence`：来源 ID、字段、操作和整数常量。核验在实际业务 handler 前执行。来源由当前任务开始之前的原始读取记录生成，本轮重新读取或写入不会替换历史来源；参考答案不进入核验。

当前支持明确的最近/首次快照、截至日期、两日期的后值减前值、两个整数域相加及明确整数偏移。使用快照的 `observed_at` 判断读取版本；它不等价于任意业务事实的有效期。未记录时间、同日冲突、没有明确主体、未知运算或不匹配的写入声明会被拒绝，返回原因供模型重新检查。

核验只约束声明过的写工具，不验证任意自然语言回答，也不能替代事务、权限或所有业务约束。不支持的需求可能无法自动完成，应交由宿主制定明确策略。

## Python 接入

```python
from token_budget_lab.context_plugin import ContextPlugin
from token_budget_lab.memory import HistoryArchive
from token_budget_lab.memory_agent import run_agent
from token_budget_lab.provenance import HistoricalWriteGuard

archive = HistoryArchive("local_results/context.sqlite3")
plugin = ContextPlugin(archive, budget=9000, offload_tools=["read_log"])
try:
    # 对符合已声明历史字段写入合同的任务显式启用。
    guard = HistoricalWriteGuard(canonical_messages)
    result = run_agent(
        canonical_messages, transport, plugin, tenant="app-user", session="conversation",
        mode="recover", max_steps=6, max_recovery_rounds=3,
        extra_tools=tool_schemas, handlers=tool_handlers, write_guard=guard,
        snapshot_read_tools=["read_config", "read_report"],
    )
    canonical_messages = result["messages"]
finally:
    archive.close()
```

`tool_schemas` 必须包含核验指定的业务工具；默认合同需要 `read_config` / `write_config` 的原文与参数结构。`snapshot_read_tools` 只接受显式注册、无参数且任务内稳定的读取。宿主可以不启用 `write_guard`，用于区分外置/筛选的效果与额外核验的效果。

独立使用 `before_model` 时，宿主负责注册 `plugin.context_tools`、调用 `invoke_context` 并追加成对的 assistant/tool 消息，以及限制恢复循环。新接口不使用 `history_restore_all`，不会把大日志重新塞回完整历史。连续没有进展的恢复会失败终止，不制造成功答案。

安装与离线查看：

```bash
python -m pip install .
python examples/python_context_agent.py
```

目前仍需 Python 宿主主动接入，没有 MCP/LangChain 自动适配。

## 固定实验协议

四组分别为原生 Full、Full 加核验、上下文插件、上下文插件加核验。两组核验使用相同来源规则、工具参数和业务操作，避免把额外校验的收益全部归因于压缩。

数据包括八个新合成实例、每实例三个连续检查点。更新、时序和拒答沿用已有任务模板；另外加入必须从当前大日志内部恢复字段的操作任务。新实例不是自然任务独立确认。中间错误不重置，评分同时检查最终回答、实际文件状态、必要工具和禁止写入的约束。

每个实例/组采用独立的首部实验标识，降低此前请求和其他组预热造成的混淆；无法强制清空服务端缓存，因此报告实际缓存字段，不把首请求直接宣称为纯冷缓存。每个完整任务后额外原样重放首次请求一次，只测缓存，不执行业务工具、不增加准确率样本，重放费用单列并计入研究总调用。

价格按已核对的 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/) 固定非高峰档重估：每百万 token 的命中输入 $0.003、未命中输入 $0.15、输出 $0.6。高峰档为两倍；重估不是到账账单。缺失或不一致的缓存分项不填零费用；未知 usage 的尝试单独保留。

```bash
python -m token_budget_lab.context_experiment --config configs/context_plugin_deepseek.json --plan
```

真实执行需本机环境变量 `DEEPSEEK_API_KEY` 和新的 `--output` 目录，配置上限为 608 次尝试，包含缓存重放。结果与限制见 [v0.6 实验报告](../results/context_plugin/REPORT.md)。
