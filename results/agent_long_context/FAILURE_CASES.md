# 失败案例：恢复循环与累计成本

本页依据固定实现对照的逐请求日志，保留失败结果。主指标仍是整个任务的答案、文件状态和必需操作，不因实际写入了一部分内容就判定成功。

恢复工具函数请求共 43 次，其中 23 次发生在恢复工具已关闭的请求中。宿主拒绝未注册工具，模型请求和后续错误处理仍计入成本。关闭工具 schema 不等于模型绝不会再次生成同名函数请求。

## temporal-03-long / 检查点 0

| 步骤 | 发送模式 | 原因 | 模型请求的工具 | 恢复工具是否注册 | 输入 token |
|---:|---|---|---|---|---:|
| 0 | compressed | selective_history_with_original_roles | history_search, read_log, read_log, read_log, read_report | True | 1,890 |
| 1 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search, history_search | False | 41,707 |
| 2 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search, history_search | False | 41,841 |
| 3 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 41,971 |
| 4 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 42,036 |
| 5 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 42,100 |

最终状态为 `step_limit`，答案正确 `False`，文件状态正确 `False`，必需工具完成 `False`。

## temporal-05-long / 检查点 0

| 步骤 | 发送模式 | 原因 | 模型请求的工具 | 恢复工具是否注册 | 输入 token |
|---:|---|---|---|---|---:|
| 0 | compressed | selective_history_with_original_roles | history_search, history_search | True | 1,890 |
| 1 | compressed | selective_history_with_original_roles | history_restore | True | 3,036 |
| 2 | compressed | selective_history_with_original_roles | read_config | True | 3,215 |
| 3 | compressed | selective_history_with_original_roles | history_restore | True | 3,287 |
| 4 | compressed | selective_history_with_original_roles | history_restore_all | True | 3,386 |
| 5 | fallback_full | agent_requested_full_history | write_config | False | 26,324 |

最终状态为 `step_limit`，答案正确 `False`，文件状态正确 `True`，必需工具完成 `False`。

## temporal-03-medium / 检查点 0

| 步骤 | 发送模式 | 原因 | 模型请求的工具 | 恢复工具是否注册 | 输入 token |
|---:|---|---|---|---|---:|
| 0 | compressed | selective_history_with_original_roles | history_search, read_log | True | 1,662 |
| 1 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search, history_search | False | 9,558 |
| 2 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 9,703 |
| 3 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 9,771 |
| 4 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 9,837 |
| 5 | fallback_full | protected_or_recovered_history_exceeds_budget | history_search | False | 9,901 |

最终状态为 `step_limit`，答案正确 `False`，文件状态正确 `False`，必需工具完成 `False`。

## 解释与后续验证

- 一例中模型主动读取多个大型诊断文件，当前工具链受保护，因此触发字符上限回退。之后仍请求已经关闭的历史工具，耗尽六步限制。
- 另一例先搜索、恢复、重读当前配置，再恢复相同记录；在第六步才写配置，尚未完成检查和最终回答。写入正确也不能通过完整任务。
- 这类失败属于恢复循环和执行控制问题，不能仅用来源消息保留率解释。进一步验证应抑制重复恢复，明确告知回退后已提供完整历史，并让必要当前输出与历史预算协调；这些改进尚未形成新的真实对照结果。
- Full 也出现错误数值、错误状态传递，以及将原始工具标记输出为文本而非可执行工具调用的情况。没有把 Full 当作参考答案，评分仍依据任务目标。
