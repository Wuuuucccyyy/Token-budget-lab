# Python Agent 历史压缩与恢复

## 目标与当前状态

目标是在原有 agent 的模型调用前筛选历史消息，并在缺少信息时恢复原文。验收顺序是任务正确性、累计输入输出 token、端到端速度。回答与完整历史版本的一致性另行统计：两个版本答得一样，并不说明都答对了。

v0.5 提供可安装的 Python 中间件、SQLite 原文归档、调用前钩子、历史恢复工具和 DeepSeek 参考 agent。在显式事实及时间范围选择的基础上，新增恢复去重、全量回退锁定、恢复轮次与业务步骤预算，以及可选的稳定只读工具控制。当前仍是实验接口。早期 24 题试验总 token 仅减少 0.18%；v0.3 工具试验全部回退 Full，总 token 增加 20.2%。长历史设置见 [协议](LONG_CONTEXT_EVALUATION.md)，循环控制结果见 [新报告](../results/recovery_guard/REPORT.md)，旧结果保留在 [v0.4 报告](../results/agent_long_context/REPORT.md)。

首版的压缩方式是删除部分旧 assistant/tool 消息，不生成有损自然语言摘要。全部原始 user/system/developer 消息、最近两轮和恢复的整轮工具记录仍按原角色、原顺序发送。这样能直接保留用户要求与更正，但用户消息很长时压缩空间有限。

## 安装与运行

Python 3.10+，核心接口仅依赖标准库。在仓库目录安装：

```bash
python -m pip install .
python examples/python_memory_agent.py
```

第二条命令仅检查筛选结果，不请求模型、不测回答质量。构建 wheel 可使用 `python -m pip wheel . --no-deps --wheel-dir local_results/dist`。wheel 包含 Python 接口，实验数据、配置和示例从源码仓库运行；尚未发布到 PyPI。

本机设置 `DEEPSEEK_API_KEY` 后，显式启动真实模型：

```bash
python examples/python_memory_agent.py --live --mode full
python examples/python_memory_agent.py --live --mode recover
```

每次示例最多三次模型调用，会产生费用；两个命令并非完整实验。示例默认使用完整历史，恢复模式需显式选择。凭证不写入归档、结果或仓库。

## 接入已有 Python Agent

宿主保存完整的 `canonical_messages`；`PreparedHistory.messages` 只作为一次模型请求的发送视图。完整状态必须追加模型回答、工具返回和后续用户请求，不得被发送视图覆盖。

```python
from token_budget_lab.memory import HistoryArchive, MemoryPlugin
from token_budget_lab.memory_agent import DeepSeekChat, run_agent

archive = HistoryArchive("local_results/history.sqlite3")
plugin = MemoryPlugin(archive, budget=9000, recent_turns=2)
try:
    result = run_agent(
        canonical_messages,
        transport=DeepSeekChat(model="deepseek-flash", max_tokens=1024),
        plugin=plugin,
        tenant="my-app-user-1",
        session="conversation-1",
        mode="recover",  # 实验模式；完整历史对照使用 full
        max_steps=6,
        max_recovery_rounds=2,
        reserve_task_steps=3,
        extra_tools=my_tool_schemas,
        handlers=my_tool_handlers,
    )
    canonical_messages = result["messages"]
finally:
    archive.close()
```

`my_tool_schemas` 是 Chat Completions function tools 列表；`my_tool_handlers` 是名称到本地可调用函数的映射。宿主负责工具权限和业务约束。恢复原文不执行历史工具，只重放已有消息记录。

可通过 `snapshot_read_tools=["read_config"]` 声明无参数且在本任务内稳定的读取工具。默认为空。参考循环读过后暂时移除该工具，其他成功业务操作完成后清空读取快照并重新开放；如果模型仍请求关闭的读取，返回已有结果并计量该轮，不再次执行。具有参数、会异步变化或有副作用的工具不能这样声明。连续两轮只请求关闭的工具会以 `tool_stalled` / `recovery_stalled` 失败终止，不能当作任务成功。

也可仅使用 `plugin.before_model(tenant, session, canonical_messages, mode="recover")` 接入自有调用循环。宿主只在 `prepared.mode == "compressed"` 时注册 `HISTORY_TOOLS`，Full 及全量回退请求不注册恢复工具。调用 `memory_tool` 后追加成对的 assistant/tool 消息，再次准备请求。恢复请求只对当前用户轮次生效，后续轮次重新检索。

`before_model` 单独负责视图选择，不会替宿主限制循环。采用自有循环时，需要实现与 `run_agent` 相同的恢复轮次计数、发送视图状态通知、全量锁定与工具停滞处理。`after_response` 在业务工具执行前调用，用于先保存已付费 usage；`after_tools` 另行记录控制事件，避免工具执行中断导致已知用量丢失。

`DeepSeekChat` 是非流式同步参考传输层，使用官方 Chat Completions 地址、关闭思考模式，不自动重试；自定义传输层返回 assistant message、usage 和 finish_reason，并以 `last` 保留已知 usage。当前没有 LangChain/MCP 自动接入，也没有与任意 agent 无缝兼容的承诺。

原生基线使用 `plugin=None, mode="full", memory_tools=False`，不注册历史工具。`DeepSeekChat(..., json_output=True)` 可启用 JSON Output，但仍需校验 answer 字段。历史实验保留原设置，没有据新格式选项重写旧分数。

## 数据流

```mermaid
flowchart TD
    A[宿主完整消息历史] --> B[按租户和会话追加原文归档]
    B --> C[保留指令 用户更正 最近轮次]
    C --> D[按主体 字段 时间范围选择原始整轮]
    D --> E{证据需求不支持或预算不足}
    E -->|是| F[完整历史回退]
    E -->|否| G[构建一次请求的发送视图]
    F --> K[关闭恢复工具]
    K --> H[模型调用]
    G --> H
    H --> I{请求恢复工具}
    I -->|是| J[绑定当前会话 搜索或恢复原文]
    J --> A
    I -->|否| K[返回回答 记录累计 usage 和耗时]
```

归档按消息位置存储原文及 SHA-256，拒绝同会话历史缩短或已有消息修改。工具消息必须有对应调用；待处理的工具调用需要全部返回后才能发起下一次模型请求。删除和恢复都以用户轮次为组，恢复到原来的角色和顺序，不把旧工具内容提升为 system 指令。

`history_search(query)` 返回整轮 ID、分数、原始工具短预览及 `already_visible`；当前任务轮次不参与搜索。`history_restore(turn_ids)` 请求下次调用注入原始整轮，返回新增、已可见、已请求 ID 和 `progress`；重复调用不再次注入。`history_restore_all()` 请求完整历史回退。模型不传入租户/会话参数，这些由宿主绑定。预览是历史数据，状态通知仅说明可用性，二者不能作为事实正确性的证明。

参考循环默认最多使用两个模型响应轮次进行恢复；一轮可以含多个函数请求，但仍受总调用步数限制。相同参数的重复搜索、未新增证据的恢复和非法恢复请求触发一次完整回退，本题后续请求保持完整历史。发生恢复后，剩余模型调用达到 `reserve_task_steps` 时停止继续恢复；默认保留 3 步，分别供业务操作、检查和最终回答使用，具体任务可能需要更多步骤。全量视图加入状态通知，明确所有原文可见及历史工具关闭。通知不写入规范历史，其输入开销仍计入实际 usage。

默认 `selector="structured"` 从实际工具 JSON 中推导主体、字段及日期，支持显式截止日期、两个时间点和完整版本范围。相同日期的冲突记录一起保留。无法结构化识别时仍走词面检索或完整回退；`selector="lexical"` 可单独使用词面选择。规则不是校准后的正确概率，也不能识别所有指代、否定或跨轮依赖。

## 预算与成本

默认预算 9,000 个序列化消息字符，包含选中消息和归档通知；它不是模型 token 上限，也不含 function tools 的 schema。调用总输入、工具 schema、恢复后的重复输入及输出均由 API usage 计量。

该阈值作用于 `before_model` 的筛选视图。参考循环追加的可用性与步骤状态通知另计入 `sent_units` 和 API usage，可能使实际发送字符超过阈值；因此它仍是软预算，不是最终请求长度保证。

预算是软上限：保护内容或恢复内容放不下时使用完整历史。若完整历史本身超过模型窗口，宿主仍需拒绝请求或采用其他策略；本插件不会通过截断必要内容强行满足限制。恢复多一次意味着又支付一次模型调用，初始历史变短不能等同于最终成本降低。

## 质量验收

主质量指标默认允许下降为 0；应用应在验证前明确绝对正确率、关键约束错误容忍度，以及统计上的可接受误差。当前小样本不能证明无损，也不宣称非劣效成立。

后续验收需要固定独立测试会话、多轮任务和重复运行：分别统计事实正确性、输出格式、任务完成率、与完整历史的答案一致性、约束违背，以及按会话聚类的配对质量差区间。仅观察“来源消息存在”不能代替生成评分；基线本身答错时应以参考答案或盲评为准。自动评分与人工语义复核单独报告。

只有质量达到预设要求的配置才比较总 token 和 P50/P95。统计必须覆盖搜索、恢复、重答、失败请求、压缩及索引耗时，不能只统计首次调用。当前真实试验保留严格格式评分，并提供明确标注为事后诊断的裸值评分，不据后者改写主结果。

## 已知边界与下一步

- SQLite 当前为单进程同步归档，不支持多写者、异步流式调用和事务性的工具执行。会话键隔离不是文件加密或外部服务的访问控制。
- 全部用户消息受保护；没有事实摘要和自动过期管理。历史增长时归档扫描、索引重建和内存索引也会增加开销。
- 目前主要检索显式标识符；中文指代、同义表达、隐藏依赖、用户更正的时间语义和恶意历史指令都需要专项测试。
- 早期试验只有 4 个开发会话、24 题；新版扩展长度条件及实际工具执行，但任务仍来自合成模板。尚未覆盖生产自然流量、任意代码修改和复杂隐藏依赖。

已加入 JSON 输出、恢复工具动态注册和任务内证据边界。后续使用未参与诊断的自然轨迹验证质量，不能把诊断后重新执行的同批任务作为独立确认。进一步扩展指代、同义表达与依赖识别时，保留原始整轮恢复及完整历史回退。

框架对接应优先采用宿主的模型调用前钩子。LangChain 的 [middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) 可作为后续集成点；[MCP tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) 可提供历史检索和恢复服务，但单独的 MCP 工具不能自动改写宿主发给模型的全部历史。记忆评测可参考 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)，当前已完成 Oracle 子集导入验证，尚未运行官方生成评审；新增原创工具评测见 [协议](AGENT_TOOL_EVALUATION.md)。
