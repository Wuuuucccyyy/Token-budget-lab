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

该示例只操作临时沙盒，最多六次模型请求。使用 `--mode full` 对照同样带有执行检查的完整历史。

## 接入 Python agent

```python
from token_budget_lab import ActionContract, RequiredAction, ContextPlugin
from token_budget_lab import HistoryArchive, run_agent

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

try:
    result = run_agent(
        canonical_messages, transport, plugin,
        tenant="user", session="conversation",
        max_steps=6, max_recovery_rounds=3,
        extra_tools=tool_schemas, handlers=tool_handlers,
        action_contract=contract,
    )
    canonical_messages = result["messages"]
finally:
    archive.close()
```

`transport(messages, tools)` 返回标准 assistant 消息及实际 usage；`tool_handlers` 执行宿主注册的业务操作。合同由宿主明确声明，尚未提供通用自然语言自动提取。完整接入见 [接口说明](docs/INTEGRATION.md)。

## 当前验证

v0.9 完成十二个长历史合成情境、三组、共 172 次真实 DeepSeek 请求。三组均带相同执行检查，条件题只声明必需读取、由原文决定写入分支；六个情境为已有诊断，六个新增数组打乱、稀疏编号、长对象、重复/缺失事件和文本末尾目标。任务成功同时要求答案、文件状态与实际操作正确完成。

| 指标 | Full | 原恢复方案 | 精准定位 |
|---|---:|---:|---:|
| 完整任务通过 | 11/12 | 4/12 | 10/12 |
| 指定本轮读取完成 | 12/12 | 12/12 | 12/12 |
| 恢复调用 | 0 | 42 | 16 |
| 输入输出总 token | 1,575,528 | 361,148 | 289,290 |
| 固定参考价格估算 USD | 0.070851 | 0.014759 | 0.012065 |
| 整任务 P95 秒 | 5.185 | 6.182 | 6.381 |

精准定位在新增六个边界情境全部通过，消除了本轮恢复停滞。相对原方案，总 token 减少 19.9%，P95 仍高 3.2%；相对 Full，总 token 减少 81.6%，固定价格费用估算减少 83.0%，P95 高 23.1%。剩余两个失败均正确读取字段后错误格式化数值，任务质量仍低于 Full，尚未达到近乎无损和更快目标。此前 169 次协议诊断单独计入，两轮共 341 次调用。全部任务与费用见 [当前评测](results/current/REPORT.md)。

默认 `evidence_mode="structured"` 提供精准定位；`evidence_mode="lexical"` 保留原恢复行为用于对照。JSON 按实际 event 字段找位置，较大事件只返回必要原文字段与路径，不假设事件编号等于数组下标。

## 下一轮重点

1. 核验写入值与原文字段的变换，避免把 409→4.09、204→2.04 写错。
2. 完善文本检索的参数提示及按来源搜索，减少工具参数错误和错误的 JSON 查询。
3. 按未完成的业务操作分配恢复预算，保留写入、检查和最终回答的请求机会。
4. 扩展当前任务来源与时序规则，覆盖相似版本及历史截至日期需求。
5. 扩充独立任务和运行次数，同时报告准确率、完整任务成本与延迟。

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
