# 实现流程

实现围绕规范历史、发送视图、恢复调用和业务执行四个环节展开。核心安装无必需第三方依赖。

| 文件 | 职责 |
|---|---|
| `memory.py` | 消息编码、工具协议检查、会话作用域与 SQLite 归档 |
| `context_plugin.py` | 大结果外置、任务内固定视图、有界恢复 |
| `retrieval.py` | 分块与词面检索索引 |
| `evidence_locator.py` | JSON 对象精确匹配、字段路径、重复计数、命中中心文本片段 |
| `history_evidence.py` | 从真实工具原文识别主体、字段和观察日期，选择历史来源 |
| `action_contract.py` | 本轮操作依赖、执行凭据与完成状态 |
| `provenance.py` | 配置历史版本及数值写入核验 |
| `write_transform.py` | 本轮实际读取绑定、字段选择、短原文投影、Decimal 重算与写入拦截 |
| `transform_tasks.py` | 冻结任务措辞到显式数值规则的有限适配，不读取评分标签 |
| `memory_agent.py` | 模型请求、工具分发、循环预算、结束检查与累计 usage |
| `langchain_model.py` | 文本消息与工具 ID 的转换、现有 DeepSeek transport 的 BaseChatModel 包装 |
| `langchain_adapter.py` | LangChain 请求视图、跨线程归档、实际执行凭据、字段权限与条件完成 |
| `langchain_experiment.py` | 单个 create_agent 图的开关对照、独立沙盒与完整调用核算 |
| `cache_cost.py` | 验证缓存分项，再按固定参考价格估算费用 |
| `action_tasks.py` / `tool_tasks.py` | 冻结合成任务与真实本地文件工具沙盒 |
| `action_experiment.py` | 当前三组对照、十步上限、按模板聚类的重复运行与全部日志 |
| `scripts/export_current.py` | 导出所有任务、请求，并从实际 handler 重新核对写入值 |
| `tool_experiment.py` / `analysis.py` | 完整任务、压缩子集和统计汇总 |

## 请求准备

`before_model` 先检查规范消息并归档，然后将宿主声明的大结果替换为来源入口。当前 user 任务第一次准备时选择历史；后续请求使用同一旧前缀，只追加新工具结果。软阈值不覆盖关键约束。

## 证据恢复

`invoke_context` 根据来源 ID 从当前作用域查原始工具结果。context_find 从实际 JSON 树推导位置，不根据事件编号猜数组下标；小对象完整返回，较大对象提供指定字段及位置。重复匹配保留计数与候选，返回长度受限。

JSON pointer 返回精确字段；自由文本检索围绕命中词截取片段并保留原文偏移。检索只索引原始业务输出。恢复结果与原始业务读取分开，避免把旧证据当作新执行。

已声明证据需求时，FreshWriteGuard.evidence_views 从实际本轮来源生成短原文字段。run_agent 在下次准备请求时传给 ContextPlugin.before_model，外置视图核对原文指纹后附上 requested_evidence。该路径不调用额外模型，也不改变规范消息。来源视图在首次发送后固定。

## 业务执行

`run_agent` 先保存模型响应 usage。模型提出业务操作后，执行合同检查前置依赖，再执行已接入的写入核验，随后才调用 handler。实际读取返回由 observe 绑定到本轮来源；历史恢复与缓存回显不经过此入口。写入被拒绝时返回结构化反馈，原值、重算值和提交值进入核验日志，实际写 handler 不执行。

当前 `format_decimal` 保留数值大小，`add_integer` 使用显式偏移。选择规则由宿主负责，unique_event 根据真实数组定位，text_record 使用明确的记录边界。规则提示不携带评分答案。每次 run_agent 复制新的核验状态，避免跨任务延续旧来源。

conditional_read_only 根据实际来源选择建立动作授权，branch_state 区分等待读取、唯一、缺失、重复与格式未解。分支变化后，run_agent 保存仅用于本轮请求的反馈，锚定在完整工具调用组之后；它不进入规范消息，也不改变已发送的前缀或工具定义。数字写入 schema 与描述将 UNKNOWN 明确排除，避免最终答案与报告参数混淆。

模型准备输出最终答案时，宿主检查所有必需操作。缺少凭据会追加遗漏反馈，继续消耗原有步骤预算；不会给未完成任务生成成功状态。

## 评测隔离

LangChain 的 before_agent 根据调用上下文建立独立状态。wrap_model_call 使用完整 graph state 生成压缩视图，wrap_tool_call 在实际 handler 前后核验授权并绑定真实读取；after_model 检查适用的必需操作，after_agent 汇总调用并关闭归档工作线程。首版不从任意自然语言自动生成核验规则。

字段权限限制的是明确声明的工具参数名，数值规则限制的是来源与变换，两者分别执行。条件操作只有在实际来源为 missing/ambiguous 时豁免；不把豁免记作执行完成。评分仍单独检查文件状态、最终答案和模型是否尝试了禁止写入。

推理输入是用户要求、原始历史、真实业务返回以及结构化操作要求。答案、目标文件状态和评分标签只在执行结束后进入 assess。测试用脚本 transport 用于协议与文件行为验证，其结果不计作大模型准确率。

`requests.jsonl` 在每次响应后落盘，包含已知 usage；工具、执行检查和任务结果分别记录。summarize 保留失败和全部请求；缓存分项不足时费用为空，不补零。
