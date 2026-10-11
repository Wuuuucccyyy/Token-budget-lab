# 连续记忆与工具执行评测

## 评测目标

本评测参考 [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 的知识更新、时序推理和拒答能力，增加实际文件读写、CSV 计算及本地检查。任务不仅要求回答正确，还要求最终文件状态正确、必需工具执行成功、用户约束没有被违反。

原创工具任务与官方 LongMemEval 数据分别处理。原创任务不称为 LongMemEval 成绩；官方 Oracle 子集目前仅完成导入验证，没有运行官方生成与评审。

## 原创任务

`data/agent_tool_tasks.json` 含 24 个任务，每个连续经过三个检查点，共 72 个检查点。四类任务各六个，开发/测试按任务 ID 隔离，各 12 个任务、36 个检查点。同一任务的三个检查点共用完整历史和文件状态，错误会传递到下一阶段；不会在评分后自动修复状态。

| 类型 | 操作与判断 |
|---|---|
| 知识更新 | 先采用最近读取值，再撤回为首次快照值加 5；必须写配置并检查 |
| 时序 | 分别采用两个指定日期之前最近的快照值；不能用现在的文件替代历史 |
| 拒答 | Alpha 没有邮箱时不得推断或写报告；Beta 的邮箱不能替代；用户明确补充后才写本地报告 |
| 工具执行 | 根据旧订单算金额并写报告，再更正币种、保留金额，最后读取报告 |

初始化创建原创虚构文件，按固定操作实际读取配置、CSV 和四份诊断日志形成历史。初始读操作由脚本执行，没有初始化阶段的模型调用；真实试验从后续检查点开始，由模型选择并执行工具。因此这是可重复的历史回放加连续执行评测，不是生产环境采集的完整 agent 轨迹。

诊断日志为生成的事件记录，用于增加历史长度；题型模板复用且每类主要改变数值。72 个检查点不能视为 72 个独立任务，也不代表自然请求流量。

## 工具与评分

工具读取或修改当前任务独立目录中的固定文件：`read_config`、`write_config`、`read_orders`、`read_log`、`read_report`、`calculate_total`、`write_report`、`run_checks`。金额用 Decimal 计算；`run_checks` 通过子进程实际执行标准库 unittest。

本地检查只验证配置正整数、报告字段和金额/币种等一般合法性，不提供隐藏参考答案。工具参数只允许操作预定文件；没有发送邮件、部署、任意路径读写或任意 shell 工具。

一个检查点通过需要同时满足：

1. 模型正常结束，最终 JSON 为 `{"answer":"值"}`，答案精确匹配参考值。
2. 实际配置、报告状态符合当前目标；标记为不写入的阶段文件保持不变。
3. 要求的业务工具实际成功执行；要求检查时最后一次检查通过。
4. 没有改变受保护配额，也没有在未授权阶段尝试写入。

参考答案和期望状态仅在执行后进入 `assess`，不传给模型、记忆检索器或本地检查程序。仅输出正确数值、未执行要求的工具，不能通过任务。

## 对照与调用记录

`full` 是原生完整历史基线：不创建插件发送视图、不注册历史工具。`compact` 删除旧工具记录，不允许恢复。`recover` 使用历史插件，并提供恢复工具。三组业务工具一致，插件工具额外占用的 schema、调用和输出全部计量。

每个条件使用独立的文件目录，相同初始数据。方法顺序按种子随机化；单个任务的检查点保持顺序。宿主持有完整历史，下一检查点追加在上一步返回的完整消息上。

每次模型调用写入 start/response 事件，业务工具执行另有审计事件。调用上限在每次发送前检查，不自动重试；中途达到上限的已知 usage 仍计入汇总。已有输出目录拒绝重放。

端到端时间包括历史准备、模型调用、业务工具和工具审计写入，排除评分、请求前后日志写入、文件初始化与初始历史回放。单轮延迟受网络和服务商缓存影响；没有测首 token、并发或完整冷启动。所有恢复调用的输入/输出计入任务总 token。

## 运行

```bash
python scripts/prepare_tool_tasks.py

# 离线：脚本策略只读取可见消息并调用真实工具，无 API
python -m token_budget_lab.tool_experiment --config configs/agent_tools_offline.json --output local_results/tools-offline

# 查看计划，无 API
python -m token_budget_lab.tool_experiment --config configs/agent_tools_deepseek.json --output local_results/tools-live --plan

# 本机设置 DEEPSEEK_API_KEY 后执行，会产生费用
python -m token_budget_lab.tool_experiment --config configs/agent_tools_deepseek.json --output local_results/tools-live --export local_results/tools-live-export
```

离线 `ScriptedPolicy` 是专用于这些任务模板的调试策略，没有访问参考标签；它用于检查消息、恢复与文件工具的连接。离线汇总的任务准确率和 token 为 null，`diagnostic_driver_success` 只表示脚本驱动的检查点通过数，不能解释为大模型能力。

## 官方 LongMemEval 数据接入

官方 [cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) 标注 MIT。当前固定版本为 `98d7416c24c778c2fee6e6f3006e7a073259d48f`，已下载 Oracle 文件并按固定种子导入更新、时序、拒答各 10 题。原始数据和转换后的完整历史保存在本地忽略目录；公开来源指纹和导入验证见 [导入记录](../results/agent_tools/longmemeval_import/validation.json)。

```bash
python -m token_budget_lab.longmemeval_adapter --input data/raw/longmemeval/longmemeval_oracle.json --output local_results/longmemeval/oracle_subset.json --variant oracle --limit 30
python scripts/validate_longmemeval.py --data local_results/longmemeval/oracle_subset.json --source-manifest data/raw/longmemeval/source_manifest.json --output local_results/longmemeval-validation
```

导入器按日期整理历史，保留 role/content 和会话日期，移除模型输入中的 `has_answer` 等标注，参考答案和来源位置只用于评测。拒答任务另保留官方参考描述，当前适配器将期望拒答值规范为 UNKNOWN；这与官方语义评审的评分口径不同。

Oracle 已提供正确证据会话，不能用它宣称现实检索效果。适配器也支持 S/M JSON 文件；正式生成应固定相应数据版本和题目 ID，并使用明确的评审规则。当前没有官方准确率或排行榜成绩。
