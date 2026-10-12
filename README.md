# Token Budget Lab

[![tests](https://github.com/Wuuuucccyyy/Token-budget-lab/actions/workflows/test.yml/badge.svg)](https://github.com/Wuuuucccyyy/Token-budget-lab/actions/workflows/test.yml)

用于 Python agent 的嵌入式上下文插件，可通过自带运行器或 LangChain 中间件接入：完整保存历史，将大工具结果外置，按当前任务恢复必要证据，并检查用户指定操作是否实际执行。目标是优先保持任务正确率，再减少完整任务的 token 消耗、实际费用和耗时。

## 当前实现

| 模块 | 作用 |
|---|---|
| `ContextPlugin` | 外置大工具结果，保留用户要求与必要记录，固定单个任务内的历史视图 |
| `HistoryArchive` | 按租户和会话保存不可改写的原始消息，检查工具调用链及原文完整性 |
| `context_find` / `context_read` / `context_search` | 按事件值查完整对象与真实字段位置，或恢复精确原文及命中中心片段 |
| `ActionContract` | 核对本轮实际执行凭据，检查指定参数与操作顺序，拦截漏操作的最终答案 |
| `FreshWriteGuard` / `WriteTransform` | 将数值写入绑定到本轮实际读取，核验格式化、整数偏移与指定参数 |
| `conditional_read_only` | 从实际读取建立本轮条件授权，区分拒答文本与数字写入，三组共享规则 |
| `eager_evidence` | 按已声明的需求直送实际读取的短原文字段，减少额外恢复请求 |
| `HistoricalWriteGuard` | 针对已声明的配置字段，核验历史快照、截止日期及数值运算 |
| `run_agent` | 将插件接到模型调用和实际业务工具之间，限制恢复循环并累计全部调用成本 |
| `TokenBudgetMiddleware` | 在同一个 LangChain agent 中切换上下文优化，保留规范历史、实际工具凭据、写入字段权限及条件完成要求 |

宿主保留完整消息，插件只改变发送给模型的视图。默认仅外置超过 1,800 字符的 `read_log` 结果；9,000 字符为软阈值。配置、订单、报告和检查结果直接保留。来源恢复不会重新执行业务工具，也不能替代用户明确要求的本轮读取。

## 快速运行

Python 3.10+，核心运行只依赖标准库。

```bash
python -m pip install .
python -m unittest discover -s tests
python examples/python_agent.py
```

示例默认检查历史视图和执行合同，不请求模型。真实执行从本机环境变量读取 `DEEPSEEK_API_KEY`：

```powershell
$env:DEEPSEEK_API_KEY = "在本机设置的密钥"
python examples/python_agent.py --live --mode context
```

该示例只操作临时沙盒，最多十次模型请求，提前完成即停止。使用 `--mode full` 对照同样带有执行与写入值检查的完整历史；`--without-eager-evidence` 保留按需恢复作为对照，`--without-write-check` 关闭数值核验。

## 接入 Python agent

```python
from token_budget_lab import ActionContract, RequiredAction, ContextPlugin
from token_budget_lab import HistoryArchive, FreshWriteGuard, WriteTransform, run_agent

archive = HistoryArchive("local_results/history.sqlite3")
plugin = ContextPlugin(archive, offload_tools=["read_log"])

# 来自当前用户要求：实际读取 scan_c，再生成报告并检查。
contract = ActionContract([
    RequiredAction("read", "read_log", {"name": "scan_c"}),
    RequiredAction("write", "write_report", after=("read",),
                   result_matches={"written": True}),
    RequiredAction("check", "run_checks", after=("write",),
                   result_matches={"passed": True}),
], before_tools={"write_report": ["read"], "run_checks": ["write"]})

# 依据任务要求构造选择与变换，不填目标值。
guard = FreshWriteGuard([WriteTransform(
    "read_log", {"name": "scan_c"}, "write_report", "total",
    {"kind": "unique_event", "array": "/events", "match_key": "event",
     "match_value": "scan_c-39", "field": "status_code"},
    fixed_arguments={"currency": "USD"},
)])

try:
    result = run_agent(
        canonical_messages, transport, plugin,
        tenant="user", session="conversation",
        max_steps=10, max_recovery_rounds=3,
        extra_tools=tool_schemas, handlers=tool_handlers,
        action_contract=contract, write_guard=guard, eager_evidence=True,
    )
    canonical_messages = result["messages"]
finally:
    archive.close()
```

`transport(messages, tools)` 返回标准 assistant 消息及实际 usage；`tool_handlers` 执行宿主注册的业务操作。合同由宿主明确声明，尚未提供通用自然语言自动提取。完整接入见 [接口说明](docs/INTEGRATION.md)。

## 接入 LangChain

v0.13.1 提供可选的 LangChain 适配。安装后可运行默认不请求 API 的接入示例：

```bash
python -m pip install ".[langchain]" -c configs/langchain_constraints.txt
python examples/langchain_agent.py
python examples/langchain_agent.py --without-plugin
```

同一个 `create_agent` 图通过 `TokenBudgetContext.enabled` 开关上下文优化，业务工具和执行、数值、条件授权检查保持相同。支持同步与异步调用；第一版文本工具接口、字段权限、条件操作与会话管理见 [LangChain 接入](docs/LANGCHAIN.md)。

完整 LangChain 对照使用同一个 agent、十二类任务各重复三次，两组均通过 36/36，最终答案全部一致。插件总 token 减少 85.5%，平均完整任务耗时由 3.869 秒降至 3.031 秒，P95 由 4.942 秒降至 3.819 秒。本轮共 265 次真实请求，18 项接入测试与 103 项核心测试通过。全部任务、请求、来源核对及前期诊断成本见 [LangChain 接入评测](results/langchain/REPORT.md)。这些合成模板仍不足以保证任意业务任务近乎无损。

## 自带 agent 验证

v0.12 使用十二个冻结的长历史合成模板，各重复三次，三组共 432 次真实 DeepSeek 请求。三组采用相同执行、写入与条件授权、同样十步上限；按需恢复与字段直送分开对照。条件题首次只声明必需读取，实际读取后才确定允许的分支。任务成功同时要求答案、文件状态、实际操作和最终回答正确完成。

| 指标 | Full + 核验 | 原插件 + 10 步 | 字段直送 + 10 步 |
|---|---:|---:|---:|
| 完整任务通过 | 36/36 | 35/36 | 36/36 |
| 文件状态正确 | 36/36 | 35/36 | 36/36 |
| 指定本轮读取完成 | 36/36 | 36/36 | 36/36 |
| 恢复调用 | 0 | 35 | 1 |
| 输入输出总 token | 4,611,948 | 859,506 | 673,900 |
| 固定参考价格估算 USD | 0.213267 | 0.036465 | 0.033092 |
| 平均整任务秒 | 4.623 | 4.444 | 3.496 |
| 整任务 P95 秒 | 5.495 | 6.201 | 4.619 |

Full 与字段直送组三次均通过 12/12，最终答案全部一致。相对 Full，总 token 减少 85.4%，固定参考费用减少 84.5%，平均耗时减少 24.4%，P95 减少 15.9%。两组全部任务均成功，无需排除失败做速度比较。相对按需恢复插件，总 token 减少 21.6%，恢复调用从 35 降到 1。

共同修正将 UNKNOWN 明确限定为最终答案，数字工具参数不接受它；缺失或重复事件只读授权在真实读取后才反馈。三组规则相同，Full 全文保留，禁止写入尝试的严格评分不变。本轮 Full 零违规；按需恢复对照仍有一次文本题恢复停滞，失败与消耗均保留。

直送组的一次数值写入 2.04 被原文核验拦截，模型改为 204.00 后完成检查和回答；纠正及额外请求计入完整成本。此前 947 次实验费用单独保留，累计 1379 次，旧质量成绩不混入本轮。

当前仍只有十二个已用于调试的合成模板，三次重复不等于新增独立任务，不能保证任意任务始终正确或更快。全部任务、逐次重复、来源核对与费用见 [当前评测](results/current/REPORT.md)。相关项目与设计理由见 [速度设计](docs/LATENCY_DESIGN.md)。

默认 `evidence_mode="structured"` 提供精准定位；`evidence_mode="lexical"` 保留原恢复行为用于对照。JSON 按实际 event 字段找位置，较大事件只返回必要原文字段与路径，不假设事件编号等于数组下标。

## 下一轮重点

1. 冻结独立任务，覆盖多字段、多文件与开放问题，验证当前收益能否推广。
2. 对未知证据需求增加按来源查询，减少恢复循环，保留任务完成机会。
3. 扩展来源修改关系及历史时序规则，覆盖相似版本与截止日期需求。
4. 继续按完整任务核算成本与延迟，检验不同历史长度和工具结果规模。

实施顺序和验收条件见 [改进路线](docs/ROADMAP.md)。

## 项目结构

```text
token_budget_lab/   插件、归档、检索、执行与评测模块
examples/           Python agent 接入示例
configs/            当前真实评测配置
data/               当前合成任务
scripts/            当前结果导出
results/current/    最新测量摘要与全部任务核算
docs/               架构、接口、代码说明与改进路线
tests/              当前插件的回归测试
```

目前提供 Python 宿主和 LangChain 中间件接入，尚未提供 MCP、skill 封装或 PyPI 发布。
