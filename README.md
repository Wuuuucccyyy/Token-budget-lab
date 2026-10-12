# Token Budget Lab

[![tests](https://github.com/Wuuuucccyyy/Token-budget-lab/actions/workflows/test.yml/badge.svg)](https://github.com/Wuuuucccyyy/Token-budget-lab/actions/workflows/test.yml)

用于 Python agent 的嵌入式上下文插件：完整保存历史，将大工具结果外置，按当前任务恢复必要证据，并检查用户指定操作是否实际执行。目标是优先保持任务正确率，再减少完整任务的 token 消耗、实际费用和耗时。

## 当前实现

| 模块 | 作用 |
|---|---|
| `ContextPlugin` | 外置大工具结果，保留用户要求与必要记录，固定单个任务内的历史视图 |
| `HistoryArchive` | 按租户和会话保存不可改写的原始消息，检查工具调用链及原文完整性 |
| `context_find` / `context_read` / `context_search` | 按事件值查完整对象与真实字段位置，或恢复精确原文及命中中心片段 |
| `ActionContract` | 核对本轮实际执行凭据，检查指定参数与操作顺序，拦截漏操作的最终答案 |
| `FreshWriteGuard` / `WriteTransform` | 将数值写入绑定到本轮实际读取，核验格式化、整数偏移与指定参数 |
| `eager_evidence` | 按已声明的需求直送实际读取的短原文字段，减少额外恢复请求 |
| `HistoricalWriteGuard` | 针对已声明的配置字段，核验历史快照、截止日期及数值运算 |
| `run_agent` | 将插件接到模型调用和实际业务工具之间，限制恢复循环并累计全部调用成本 |

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

## 当前验证

v0.11 使用十二个冻结的长历史合成模板，各重复三次，三组共 450 次真实 DeepSeek 请求。三组采用相同执行与写入核验、同样十步上限；只增加步骤的插件与直送字段的插件分开对照。条件题只声明必需读取、由实际原文决定写入分支。任务成功同时要求答案、文件状态、实际操作和最终回答正确完成。

| 指标 | Full + 核验 | 原插件 + 10 步 | 字段直送 + 10 步 |
|---|---:|---:|---:|
| 完整任务通过 | 33/36 | 35/36 | 36/36 |
| 文件状态正确 | 36/36 | 36/36 | 36/36 |
| 指定本轮读取完成 | 36/36 | 36/36 | 36/36 |
| 恢复调用 | 0 | 43 | 5 |
| 输入输出总 token | 4,899,362 | 879,100 | 669,608 |
| 固定参考价格估算 USD | 0.213970 | 0.036523 | 0.032944 |
| 平均整任务秒 | 4.489 | 4.637 | 3.567 |
| 整任务 P95 秒 | 5.702 | 6.283 | 4.277 |

字段直送组三次均通过 12/12，最终答案与 Full 全部一致。相对 Full，总 token 减少 86.3%，固定参考费用减少 84.6%，平均耗时减少 20.5%，P95 减少 25.0%。在两组均成功的同一批 33 个任务条件中，P95 仍减少 24.9%。相对只增加步骤的插件，总 token 减少 23.8%，恢复调用从 43 降到 5。

Full 的三个失败是禁止写入任务中尝试调用写工具，宿主拦截、文件没有改动，最后答案仍正确；原插件的一次失败是缺失事件恢复停滞。失败与全部消耗均保留。直送组没有真实拒写，不能将其成绩全部解释为纠错干预。

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

目前通过 Python 宿主显式接入，尚未提供 MCP、LangChain 自动适配或 PyPI 发布。
