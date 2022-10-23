<p align="center">
    <br>
    <img src="./figure/logo.png" width="275"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/xlxwalex/FCGEC/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/xlxwalex/FCGEC.svg?color=blue&style=flat-square">
    </a>
</p>

---

# FCGEC: Fine-Grained Corpus for Chinese Grammatical Error Correction
[**中文**](https://github.com/xlxwalex/FCGEC) | [**English**](https://github.com/xlxwalex/FCGEC/blob/main/README_EN.md)

## FCGEC介绍
中文语法检错纠错（Chinese Grammatical Detection & Correction, CGED & CGEC）是指给定一个句子，检纠错系统可以检查句子中是否存在语法错误。如果有语法错误，系统需要对错误的文本进行自动纠错并输出正确的句子。
该项技术已被用在教育、检索等多个领域。

近来在数据集以及模型上都有了不少的工作，但是之前的工作主要有三个问题：(1)中文的数据集较少。 (2)中文纠错数据主要集中在汉语水平考试(HSK)等类似来源，因此语料库包含的主要是非中文母语使用(Chinese as a Foreign Language, CFL) 的错误数据，而母语使用者的语法错误要更具挑战性。 (3) 纠错的方式单一，往往只有一种修改方式。


基于以上这些问题，我们的FCGEC旨在提供一个大规模母语使用者的多参考文本纠检错语料，用于训练以及评估纠检错模型系统。除此之外，在这个工作中我们提出了一种基于编辑的（Switch-Tagger-Generator, STG）模型作为基准，希望能为CGEC社区做出贡献。

## FCGEC语料
我们的数据来源主要是小初高中学生的病句试题以及新闻聚合网站，为了给句子更多的参考修改方式来达成多样化的标注目标，每一个句子会被随机分配给2-4个标注者进行标注。我们从两个数据来源中收集到了54,026个原始句子，经过去重和筛选掉问题句(如文本截断等)后FCGEC共包含41,340个句子。数据的统计信息如下表所示：

 数据集 | 数据来源| 句子总数 | 错误句子数(%) | 平均长度 | 平均参考数 | 
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | 
| **FCGEC** | `Native` | 41340 | 22517 (54.47%) | 53.1 | 1.7 | 

我们将这些数据分成了训练集，验证集和测试集，它们的数量分别为：36340， 2000以及3000。其他更多详细的数据请参见我们的论文。

### FCGEC语料数据
FCGEC的训练、验证及测试数据都已放在`data`目录下，数据的格式请见`data`下的README文件。注意：测试集我们只给出了句子没有给出对应的标签，您可以通过该[Codalab评测页面](https://codalab.lisn.upsaclay.fr/competitions/8020) 提交您的模型预测结果来计算性能。

## FCGEC任务
我们的语料共有三种标签对应于纠检错的三个任务，分别如下：
+ ***错误检测 (error detection)：*** 模型需要判断给定的句子是否包含语法错误 (二分类任务)
+ ***类型检测 (type identification)：*** 模型需要判定句子中的语法错误属于七种错误类型中的哪一种，错误类型分别为：
    1. 语序不当 (Incorrect Word Order, IWO)
    2. 搭配不当 (Incorrect Word Collocation, IWC)
    3. 成分缺失 (Component Missing, CM)
    4. 成分赘余 (Component Redundancy, CR)
    5. 结构混乱 (Structure Confusion, SC)
    6. 不合逻辑 (Illogical, ILL)
    7. 语意不明 (Ambiguity, AM)
+ ***文本纠错 (error correction)：*** 给定一个句子，模型输入对应的无语法错误句子

关于以上任务，更多详细的信息以及实例请参考我们的论文。
