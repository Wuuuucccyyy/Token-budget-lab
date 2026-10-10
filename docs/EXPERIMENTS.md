# 实验协议与复现

## 评估优先级

任务准确率是首要目标。先检验答案是否正确、完整并得到材料支持，再比较输入/输出 token 和端到端速度。当前离线检索器的片段包含率与答案字符串保留率属于诊断，不构成真实任务准确率证据。

新版短答案 API 试验在所有策略中加入相同的简短回答指令，以 EM 为主质量指标，F1 为补充，并保留人工正确性复核。默认 `quality_metric=exact_match`、`allowed_quality_drop=0`。允许下降的值是评分尺度上的绝对差，需在查看测试结果前固定，不能依据节省率事后放宽。

汇总先对同题、同轮结果配对，再按材料聚类计算主指标差值区间。数据完整且至少包含两份材料，区间下界达到预设门槛时，才标记 `meets_configured_screen`；区间跨越门槛为 `inconclusive`，明确低于门槛为 `below_configured_screen`。离线、片段代理指标、全零基线均不能用于通过真实质量筛查。

该规则是当前材料上的相对自动评分筛查；尚需评估绝对准确率、人工盲评和任务覆盖。质量未达到要求的配置仍公开结果，但不因更省 token 或更快而列为推荐。

## 速度测量

新版 `pipeline-v2` 从缓存查询开始计时，直到答案准备完成，包含缓存范围指纹计算、查找、上下文筛选和生成。统计评测与日志写入的耗时不混入请求处理速度。逐题日志保存 `cache_lookup_seconds`、`compression_seconds`、`end_to_end_seconds`，汇总端到端均值、P50、P95，并分别统计应用层缓存命中与未命中。

API transport 耗时另外记录；当前非流式响应只测完整答案耗时，没有首 token 延迟。模型权重初始化不计入逐请求耗时，冷启动和并发吞吐量需另做实验。离线耗时也不能外推为真实 API 速度。

历史日志的计时范围围绕旧版 `Pipeline.run`，没有分阶段记录；新增 [历史速度复核](../results/latency/DEEPSEEK_LATENCY.md) 公开数值字段。历史固定顺序、单轮实验只支持描述，不支持稳定速度优势结论。后续在相同硬件、模型、提示词与输出设置下，多轮交错方法顺序，分别观察命中、未命中和长尾请求。

## 已完成与待执行

| 实验 | 状态 | 可复核材料 |
|---|---|---|
| 74 条合成请求离线实验 | 已完成 | `results/REPORT.md` |
| 2026-10-09 DeepSeek 合成数据实验 | 已完成，存在一次未计量请求 | `results/DEEPSEEK_EXPANDED_REPORT.md` 与公开 CSV |
| SQuAD Article 测试集：100 题 × 12 配置 | 离线运行完成 | `results/public/` |
| 新版 DeepSeek 公开数据试验 | 配置、计划检查及模拟接口测试完成；真实调用尚未执行 | `configs/deepseek_pilot.json` |
| LLMLingua-2 | 可选适配器完成；权重推理尚未执行 | `configs/llmlingua2_optional.json` |
| 人工答案评审 | 表格生成完成；尚未评分 | `failures.py` 生成的 `manual_review.csv` |

本次运行环境无法通过终端访问 DeepSeek 或下载模型权重，因此不报告新增 API 或 LLMLingua-2 实测数字。

上述限制描述已交付实验的运行条件；后续执行真实试验时需重新核验连接与权限，历史结果不据当前环境变化改写。

本地演示的请求处理函数已作内存传输测试；当前环境也限制 loopback 连接，尚未完成浏览器端到端验证。

## 离线一键复现

```bash
python -m unittest discover -s tests -v
python scripts/reproduce.py --output local_results/public-reproduce
```

需要图表时安装 `requirements-analysis.txt`，然后在命令末尾加 `--plots`。输出包含运行清单、逐请求日志、汇总、报告、失败案例与空白人工评审表。图表模式还从公开历史 CSV 重算材料簇 bootstrap 并生成 DeepSeek 输入/输出 token 图。

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce.py --output local_results/public-with-plots --plots
```

`configs/public_offline.json` 固定测试集、随机种子和 12 组配置。每组处理相同顺序的问题，各组缓存互相隔离。25% / 50% / 75% 是上下文字符上限；系统指令和问题不压缩。各方法实际长度同时记录，不能把相同字符比例称为相同实际模型 token 数。

## DeepSeek 公开数据试验

先复制配置并把 `model` 替换成账户实际可用模型。配置默认从 10 篇开发集文章各取一题，比较四种方法，上限 40 次调用、每次最多 512 输出 token。`--dry-run` 不需要密钥，也不调用模型。

```bash
python -m token_budget_lab.experiment --config configs/deepseek_pilot.json --dry-run
python -m token_budget_lab.experiment --config configs/deepseek_pilot.json --output local_results/deepseek-public
```

凭证只从 `DEEPSEEK_API_KEY` 环境变量读取。默认使用 Chat Completions 协议并显式关闭思考模式。`protocol=anthropic` 可切换到 DeepSeek 官方 Anthropic 兼容端点；思考开关依照 [官方参数定义](https://api-docs.deepseek.com/guides/thinking_mode/) 映射。

历史实验使用 Anthropic 协议、默认思考模式、256 输出 token；新版默认设置与其不同，不能直接把两轮差异归因于压缩方法。旧 `deepseek.py` 命令保留兼容性，新实验统一使用 `experiment.py`。

`max_calls` 限制累计尝试次数，不是金额预算。真实费用还取决于输入长度、输出、服务端缓存及当日价格；本项目不从字符数虚构费用估计。`repeats` 可执行多轮，固定种子随机化每轮策略顺序，重复运行不增加独立材料数。API 采样参数使用服务端默认值，种子只控制本地顺序，不控制模型随机性。

## 日志、恢复与错误

- `manifest.json` 保存完整配置、数据 SHA256、源代码 SHA256、题目 ID 和 Python 版本。
- `requests.jsonl` 是追加日志；发起付费请求前写入 `start`，响应处理后写入 `result`，刷新并同步到磁盘。
- `--resume` 要求配置、数据、代码一致，跳过已完成结果并重建每组缓存。输出目录不能由多个进程同时写入。
- 没有对应结果的 `start` 可能已经产生费用，因此阻止自动重放。已记录 API 错误同样不自动重试；核查服务端记录后另开运行目录。
- 空答与截断保存已知 usage，参与质量统计，但不写入回答缓存。未知 usage 明确计数，不作为零费用。
- 达到调用上限时标记 `call_limit`，异常标记 `incomplete`，均不属于完整对照实验。

## 邻域方法与消融

`bm25_neighbor` 先按 BM25 对句子排序，再将每个候选锚句及其左右邻句依次加入候选顺序，最后在严格预算内贪心保留并恢复原文顺序。它优先分配邻近句的预算，不保证锚句与所有邻句同时被选中。

半径 0 应与 BM25 完全一致，此项由全预算范围单元测试及公开数据消融共同核验；半径 1、2 检查局部上下文收益与预算竞争。半径增大可能挤掉远处的关键证据，因此保留负结果。

## LLMLingua-2 入口

```bash
python -m pip install -r requirements-llmlingua.txt
python -m token_budget_lab.experiment --config configs/llmlingua2_optional.json --output local_results/llmlingua2
```

运行前必须把 `revision` 替换为所选 Hugging Face 模型的 40 位 commit SHA。首次运行需要下载权重，CPU 也可执行但耗时取决于硬件。依赖未安装或模型不可用时直接报错，不回退成 BM25。

适配器调用 [官方 PromptCompressor](https://github.com/microsoft/LLMLingua)，保留问题不压缩；模型自身采用其 tokenizer。若输出超过外层计数器预算，额外裁去末尾字符直到满足上限。故配置名称明确标为 `llmlingua2-postfit`，这一裁剪可能损伤信息，不能等同于未经修改的官方基线。仍需在实际模型 token 预算下完成公平比较。

## 人工复核与公开产物

`manual_review.csv` 的 factual_correct、evidence_sufficient、notes 字段留空。建议评审时隐藏策略名称，对同一道题的输出打乱排序，分别判断事实正确性与证据完整性；未进行实际评分之前不报告人工正确率。

公开库保存汇总、逐题数值、图表与少量公开数据案例。完整本地日志可能较大，位于已忽略的 `local_results/`；凭证和 CC Switch 配置不进入版本库。历史材料区间只表示既定材料集合的抽样波动，不代表真实流量总体。
