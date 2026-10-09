# SQuAD Article QA：公开数据改编

本项目从 SQuAD v1.1 的公开开发集构建文章级问答任务。来源为 Pranav Rajpurkar、Jian Zhang、Konstantin Lopyrev 和 Percy Liang 发布的 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)；原始材料取自 Wikipedia。

原始文件：[dev-v1.1.json](https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json)。固定 Git blob：`e9a3f913ad1468ebe105b891334ca7b0bc0e2510`。脚本额外核验规范化 JSON 的 SHA256，拒绝不同版本。

数据与改编文件 `squad_article.json` 按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 发布；本项目代码的 MIT 许可不替代数据许可。基于这些数据的失败案例摘录与人工评审表同样保留来源与此许可。

## 改编过程

1. 按文章标题排序，再用 Python `random.Random(20261010)` 打乱文章。
2. 将每篇文章在原数据中的全部段落依次以两个换行连接。材料是数据集收录的段落集合，并非当前 Wikipedia 完整页面。
3. 在每篇文章中按问题 ID 排序，不依据答案长度、答案位置或模型效果筛选，固定抽取 10 道题。
4. 前 28 篇文章分到 train，接下来 10 篇为 dev，最后 10 篇为 test。同一文章只属于一个集合。
5. 保留全部参考答案作为可接受的替代答案；只需与其中一个匹配。保留原始段落 SHA256 作为来源校验字段。

共 48 篇文章、480 道题：train 280、dev 100、test 100。文章材料长度为 14,746–85,500 字符，排序后上中位数为 31,983 字符。数据按文档去重存储，加载时展开到问题，避免在版本库重复保存相同长材料。

这里的 train/dev/test 是本项目对原官方 dev 的重新划分，不是 SQuAD 官方划分。当前未训练任何模型；train 预留用于后续方法开发。测试集上的 12 组设置是固定对照与半径消融，未根据测试结果选择最终参数。

## 复现数据文件

从上述来源下载原始 JSON 到 `data/raw/dev-v1.1.json` 后运行：

```bash
python scripts/prepare_squad.py --source data/raw/dev-v1.1.json --output local_results/squad_article_rebuilt.json
```

比较生成文件与仓库中的 `data/squad_article.json` SHA256。样本 ID、划分与种子均写在数据文件中；实验清单另保存整个数据文件的指纹。

## 任务与指标边界

问题原本针对短段落；本项目加入同篇文章其他段落作为干扰内容。因此，这是 SQuAD 派生的文章级检索实验，不是官方榜单成绩，也不是 LongBench 复现。数据仅含可回答题，不能用于测量无答案检测。

答案字符串保留不等于完整证据保留，多个位置出现同一字符串时尤其如此。EM/F1 使用英文归一化；离线回答器返回整句而参考答案通常是短片段，因此 EM 偏低属于回答形式差异，不能解读为生成模型准确率。中文合成样例继续采用既有片段与证据指标，避免混用英文评分规范。
