# 公开数据失败案例

以下案例按运行日志顺序选择，限定 50% BM25 与邻域方法中答案片段未命中的前六个不同问题；不是随机代表性抽样。回答来自离线句子检索器。
人工评分表预留字段为空，尚未开展人工正确性评审。公开数据及其答案摘录按 CC BY-SA 4.0 使用，来源见 data/PUBLIC_DATA.md。

## neighbor1-0.5 · 572872dd2ca10214002da37e

文章：University_of_Chicago

问题：Who calculated the speed of light?

可接受答案：A. A. Michelson

检索器返回：

> Notable faculty in physics have included the speed of light calculator A.

答案原文仍在压缩上下文中：否；实际上下文比例：50.0%。

诊断：压缩后已找不到参考答案字符串；需要检查相关句排序、预算竞争和分句边界。

## neighbor1-0.5 · 572847ff3acd2414000df869

文章：University_of_Chicago

问题：In what year did student decide to occupy the president's office?

可接受答案：1962

检索器返回：

> In 1929, the university's fifth president, Robert Maynard Hutchins, took office; the university underwent many changes during his 24-year tenure.

答案原文仍在压缩上下文中：是；实际上下文比例：50.0%。

诊断：答案字符串尚在上下文中，失败发生在离线回答选择环节；不能直接归因于压缩丢失答案。

## neighbor1-0.5 · 57284142ff5b5019007da00a

文章：University_of_Chicago

问题：Who donated property to the University of Chicago?

可接受答案：Marshall Field

检索器返回：

> The University of Chicago agreed to confer a degree on any graduating senior from an affiliated school who made a grade of A for all four years, and on any other graduate who took twelve weeks additional study at the University of Chicago.

答案原文仍在压缩上下文中：是；实际上下文比例：50.0%。

诊断：答案字符串尚在上下文中，失败发生在离线回答选择环节；不能直接归因于压缩丢失答案。

## neighbor1-0.5 · 57285213ff5b5019007da182

文章：University_of_Chicago

问题：How many divisions make up the academics of the university?

可接受答案：five

检索器返回：

> The first buildings of the University of Chicago campus, which make up what is now known as the Main Quadrangles, were part of a "master plan" conceived by two University of Chicago trustees and plotted by Chicago architect Henry Ives Cobb.

答案原文仍在压缩上下文中：是；实际上下文比例：50.0%。

诊断：答案字符串尚在上下文中，失败发生在离线回答选择环节；不能直接归因于压缩丢失答案。

## neighbor1-0.5 · 57286ec63acd2414000df9d3

文章：University_of_Chicago

问题：What popular environmentalist is also a university alumni member?

可接受答案：David Suzuki, / David Suzuki

检索器返回：

> After continued turmoil, a university committee in 1967 issued what became known as the Kalven Report.

答案原文仍在压缩上下文中：是；实际上下文比例：50.0%。

诊断：答案字符串尚在上下文中，失败发生在离线回答选择环节；不能直接归因于压缩丢失答案。

## neighbor1-0.5 · 572860e03acd2414000df979

文章：Yuan_dynasty

问题：What dynasty came after the Yuan?

可接受答案：Ming dynasty / the Ming dynasty

检索器返回：

> The physicians of the Yuan court came from diverse cultures.

答案原文仍在压缩上下文中：是；实际上下文比例：50.0%。

诊断：答案字符串尚在上下文中，失败发生在离线回答选择环节；不能直接归因于压缩丢失答案。
