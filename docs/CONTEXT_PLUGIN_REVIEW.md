# Agent 上下文插件的实现对照与问题审查

审查日期：2026-10-11。依据项目官方仓库、关键源码和本项目 v0.5 固定实现日志。没有在相同 DeepSeek 任务上运行这些外部项目，因此这里比较接入机制，不把其宣传压缩率当作本项目可以达到的成绩。

## 可参考的实现

| 项目 | 接入方式与处理位置 | 可采用的设计 |
|---|---|---|
| [Headroom](https://github.com/headroomlabs-ai/headroom) | Python SDK、代理及框架适配；按内容类型压缩工具结果，原文按哈希缓存，提供恢复接口 | 小型关键事实直接保留；大块内容局部恢复；默认 cache 模式保持旧消息稳定，只处理新内容 |
| [Context Mode](https://github.com/mksglu/context-mode) | MCP 工具结合宿主 hooks；在执行和查询环节控制返回结果体积 | 原始日志留在沙盒或索引中，按意图查询返回片段，避免把整个日志先写入上下文 |
| [DeepAgents](https://github.com/langchain-ai/deepagents) | 原生模型调用和工具调用中间件；大结果外置，历史达到阈值后摘要 | 对工具结果与历史分别处理；保存原始历史和后端位置；计入系统消息及工具定义预算 |
| [RTK](https://github.com/rtk-ai/rtk) | 命令包装器、hook 或宿主插件；改写命令并整理输出 | 对已知命令采用确定性格式整理；失败原文可召回，减少重新执行 |
| [Graphiti](https://github.com/getzep/graphiti) | Python 记忆库和时序知识图谱；维护事实、实体和来源 | 事实有效期、记录时间及来源分别存储；按时间选择事实，保留被替代的历史。它不是即插即用的无损历史压缩器 |

Headroom 的 [架构文档](https://docs.headroomlabs.ai/docs/architecture) 区分 cache 和 token 模式：前者保留已经发送的旧消息，后者可以改动较早内容以减少原始 token。其 [CCR 文档](https://docs.headroomlabs.ai/docs/ccr) 使用内容哈希保存原文并支持按需取回。原文可恢复不能保证模型会正确调用恢复工具，也不能保证任务答案正确。

Context Mode 的 [server.ts](https://github.com/mksglu/context-mode/blob/main/src/server.ts) 在执行输出较大且提供 intent 时先索引，再返回匹配预览；超过更大阈值时返回索引入口。其 [README](https://github.com/mksglu/context-mode#how-the-knowledge-base-works) 仍采用 FTS5/BM25。由此不能把本项目失败简单归因于 BM25 不够先进；返回哪些证据、何时返回以及如何使用证据同样关键。

DeepAgents 的 [summarization.py](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/summarization.py) 通过 wrap_model_call 构建有效消息视图，保留原始状态并记录摘要事件；摘要需要模型调用，这部分费用应计入。其 [filesystem.py](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/filesystem.py) 提供工具结果外置及分页访问机制。选择摘要本身不构成质量保证。

RTK 的 [README](https://github.com/rtk-ai/rtk) 描述命令执行前改写和失败输出召回。Graphiti 的 [edges.py](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py) 分别记录 valid_at、invalid_at、created_at、expired_at 和来源 episodes。可以采用其时序与来源设计，暂不引入完整图数据库和自动事实抽取流程。

上述项目的部分压缩展示按工具输出字节或 token 计量；它们的模型、任务、检索次数和质量口径不同。不能将工具输出减少比例直接与本项目全任务 API usage 比较，也不能推导近乎无损。

## v0.5 中可以确认的问题

固定实现包含 6 个长历史任务、18 个连续检查点，两种模式各运行一次。插件通过 16/18，Full 通过 17/18；完整任务均通过 5/6。没有步骤耗尽或截断，仍存在错误写入。

### 大工具结果进入当前轮后无法缩小

插件全部 58 次模型请求中，34 次是压缩视图，24 次是 Full 回退。压缩请求累计输入 81,016 token，回退请求累计输入 802,655 token，占插件输入的约 90.8%。这说明主要开销集中在回退，首次压缩并不是任务成本的主要决定因素。

失败任务 temporal-05-long 的第二个检查点在正确快照已经可见时，仍读取 scan_b、scan_c、scan_d 和 report。日志原文进入受保护的当前工具链，下一请求超过预算，整段历史回退 Full。当前插件主要处理旧轮次，缺少工具结果进入历史时的体积控制。

### 正确来源存在，写入没有来源约束

截至 2026-10-06 的正确 Alpha 快照是 2026-10-05 的 timeout_ms=985，最初视图确实保留该来源。模型回退后却写入 2026-10-01 的 615。write_config 只接收整数，没有验证其来源是否符合本次时间要求；run_checks 只检查类型和正值等约束，615 同样通过。

后续检查点只汇报检查结果且禁止写文件，PASS 与实际检查一致，但错误文件仍然保留。因此两个失败检查点主要来自一次错误写入及后续状态延续，不是两个独立版本选择错误。详见 [失败轨迹](../results/recovery_guard/FAILURE_CASES.md)。

### 总 token 与费用方向不同

从原始响应 usage 重新核算：

| 指标 | Full | 插件 |
|---|---:|---:|
| 输入 token | 1,160,394 | 883,671 |
| 缓存命中输入 | 1,151,487 | 677,632 |
| 缓存未命中输入 | 8,907 | 206,039 |
| 输入缓存命中比例 | 99.2% | 76.7% |
| 输出 token | 1,597 | 2,397 |

未命中输入增加约 23.1 倍。按 [DeepSeek 当前价格](https://api-docs.deepseek.com/quick_start/pricing/) 的 Flash 非高峰档估算，Full 约 $0.00575，插件约 $0.03438，约为 6.0 倍。这是按当前公开价格重估这一轮 usage，不是到账账单或历史费用证明。没有计入套餐、赠款和账户调整。

DeepSeek 的 [缓存机制](https://api-docs.deepseek.com/guides/kv_cache/) 依赖重复前缀。我们的每次历史筛选、压缩与 Full 切换，以及在原始指令后插入变化的控制通知，会改变较早部分内容。日志可以确认命中量差异；尚不能把差异全部归因于其中某一种改写。

两个条件都使用已参与诊断的任务，Full 也可能受此前重复请求的预热影响。冷缓存和暖缓存尚未分别控制，这组数据不能推广为插件通常使费用增加六倍。当前报告中的 23.7% 总 token 下降仍然成立，但不能据此声称费用下降。机器可读核算见 [cache_accounting.json](../results/recovery_guard/cache_accounting.json)。

## 尚待验证的原因

1. 全部旧 user 请求保留，而部分旧 assistant/tool 完成记录被删除，可能使已经完成的读取指令再次被执行。需要与“保留任务完成状态”的视图作对照，不能直接删除仍有效的用户约束。
2. Full 回退会把旧值、有效值、后续派生值及无关日志一起恢复，可能干扰版本选择。需要比较局部恢复、整轮恢复和全量恢复；当前轨迹不能揭示模型内部选择旧值的具体原因。
3. 多项控制同时改变，且复查重复使用诊断场景，尚无法证明观察到的改善来自某一个机制。需固定数据，逐项消融，并在未观察任务上重复验证。

## 下一阶段的实现顺序

1. 增加工具结果钩子。完整原文仍归档；较小配置快照原样保留，大日志返回有来源的索引入口和有限片段。恢复接口支持主体、字段、时间和位置范围，避免先搜索 ID 再恢复整轮的固定两步开销。
2. 在请求模型前组成任务证据视图：当前有效约束、任务状态、必要来源和受保护的工具调用链。保留稳定的已发送视图，避免每次查询都重排整个前缀；达到阈值时再进行阶段性压缩。
3. 对历史恢复类写入增加来源声明和核验。先覆盖直接复制字段的操作，再处理多来源计算。校验依据来自实际原文及当前用户要求，不能来自参考答案。
4. 固定实验协议，分别测试冷缓存与暖缓存；计量命中输入、未命中输入、输出、恢复及摘要调用。分别报告插件无来源校验、插件有来源校验；通用业务校验也应给 Full 对照，避免把额外能力的收益全部归为压缩收益。

验收保持任务质量、实际费用及 token、端到端速度的顺序。小型专项调试任务用于找错误；新自然轨迹与预先规定的质量门槛用于判断是否适合默认启用。
