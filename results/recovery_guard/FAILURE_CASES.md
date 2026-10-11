# 保留的失败与工具轨迹

所有失败均保留。以下消息来自原创合成任务，按实际响应提取；不从解释中重评分。

## diagnostic

| 任务 | 模式 | 检查点 | 状态 | 答案 | 文件 | 必需工具 |
|---|---|---:|---|---|---|---|
| temporal-03-long | full | 0 | ok | False | False | True |
| temporal-03-long | recover | 0 | truncated | False | False | False |
| knowledge_update-04-long | recover | 0 | ok | True | False | False |
| knowledge_update-04-long | recover | 1 | step_limit | False | True | False |
| temporal-05-long | recover | 0 | step_limit | False | True | True |
| temporal-05-long | full | 0 | ok | True | False | False |

逐步响应见 [failure_trace.json](failure_trace.json) 的 `diagnostic`，工具操作与最终文件见该轮审计目录。

## fixed_implementation

| 任务 | 模式 | 检查点 | 状态 | 答案 | 文件 | 必需工具 |
|---|---|---:|---|---|---|---|
| abstention-04-long | full | 0 | ok | False | True | True |
| temporal-05-long | recover | 1 | ok | False | False | True |
| temporal-05-long | recover | 2 | ok | True | False | True |

逐步响应见 [failure_trace.json](failure_trace.json) 的 `fixed_implementation`，工具操作与最终文件见该轮审计目录。

## 固定实现中的剩余错误

`temporal-05-long` 的检查点 1 需要截至 2026-10-06 的有效快照值 985，模型却写入首次快照的 615。目标来源消息已经保留，仍发生版本使用错误。实际 run_checks 通过，但它检查类型与业务约束，不能证明选择了正确历史版本。

检查点 2 的 PASS 与实际检查结果一致，但错误的 timeout_ms 继续保留，因此文件状态评分仍失败。Full 的另一个失败来自拒答题最终格式，不据事后解析改写成绩。

后续应优先验证来源日期与本次写入值的对应关系，并控制无关日志读取；循环顺利结束不是任务正确性的充分条件。
