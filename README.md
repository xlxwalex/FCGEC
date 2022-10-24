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

近来在数据集以及模型上都有了不少的工作，但是之前的数据集工作主要有三个问题：(1)中文的数据集数量较少。 (2)中文纠错数据主要集中在汉语水平考试 (HSK) 等来源，因此语料库包含的主要是非中文母语使用(Chinese as a Foreign Language, CFL) 者的错误数据，而母语使用者的语法错误要更具挑战性。 (3) 纠错的方式单一，往往只有一种修改方式 (修改答案)。


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

## STG纠错模型
对于FCGEC的纠错任务，我们提出了STG (Switch-Tagger-Generator)模型，如下图所示。其由三个模块组成：
+ `Switch`模块：利用指针网络来确定句子中字符序列的顺序
+ `Tagger`模块：预测纠错需要对每个字符进行操作 (保持-KEEP， 删除-DELETE， 插入-INSERT， 修改-MODIFY)，并且对于插入和修改操作还需要确定操作的字符的数量T
+ `Generator`模块：当涉及INSERT以及MODIFY这样需要改变字符的操作时，利用预训练模型的MLM方式来生成字符

***注意：*** 我们的STG模型各个模块之间可以自由的`独立`或`联合`训练，但是在推断时是以Pipeline的形式进行的

<p align="center">
    <br>
    <img src="./figure/stg.png" width="300"/>
    <br>
</p>
<p align="center">
    <br>
    STG模型示意图
    <br>
</p>
在STG模型的基础上，我们发现由于错误类型与我们设计的几种操作关系有很大的相关性，因此检错任务可以通过将错误类型判断作为辅助任务来提升纠错的性能，也就是论文中的(+TTI)。

更多关于模型的信息可以参考我们的论文

### 实验环境搭建
我们采用Python=3.8.5作为基本环境，您可以用通过以下代码来创建环境以及安装依赖：
```shell
conda create -n stg_env python=3.8.5
conda activate stg_env
pip install -r requirements.txt
```

### 训练及测试
STG模型训练-测试bash文件共有三个：[`run_stg_indep.sh`](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/run_stg_indep.sh) , [`run_stg_tti.sh`](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/run_stg_tti.sh) 以及[`run_stg_joint.sh`](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/run_stg_joint.sh) ，分别对应论文中的`STG-Indep`，`STG-Indep+TTI`以及`STG-Joint`。具体细节可参考 `model/STG-correction/run_stg_indep.sh`，`model/STG-correction/run_stg_tti.sh`和`model/STG-correction/run_stg_joint.sh`。

***注意***：在使用前请先配置bash文件头部的参数：
```shell
CUDA_ID=   
SEED=                               # 随机数种子
EPOCH=     
BATCH_SIZE=  
MAX_GENERATE=                       # MAX T (最大生成字符数，一般设为6即可)
CHECKPOINT_DIR=checkpoints
PLM_PATH=                           # 预训练模型路径
OUTPUT_PATH=                        #测试集预测输出.xlsx文件位置
```

## 模型性能评测
对于***错误检测*** 以及***错误类型检测*** 两个任务，我们采用`Accuracy`, `Precision`, `Recall` 以及 `Macro F1 score` 作为衡量模型性能的依据。

对于***文本纠错任务***，我们采用了`Exact Match`以及`character-level edit metric` ([MuCGEC](https://github.com/HillZhang1999/MuCGEC) 中提出)作为评价指标。

更详细的内容请见 [`scorer`](https://github.com/xlxwalex/FCGEC/tree/main/scorer) 目录下的README文件。

### 在线评测页面
我们的测试集不直接提供三个任务的标签，因此您需要通过在线评测页面的形式提交您的测试集模型预测结果得到模型的性能指标，我们将评测页面部署在了`Codalab`上并永久开放，您可以通过下方链接进行访问：
<p align="center">
    <a href="https://codalab.lisn.upsaclay.fr/competitions/8020">
        <img alt="Codalab" src="https://img.shields.io/badge/ FCGEC- CodaLab-plastic?style=for-the-badge&logoColor=white&link=https://codalab.lisn.upsaclay.fr/competitions/8020&logo=codalab">
    </a>
</p>

## 相关数据集工作
+ MuCGEC评测数据集：[MuCGEC](https://github.com/HillZhang1999/MuCGEC/)
+ YACLC评测语料库：[YACLC](https://github.com/blcuicall/YACLC)
+ NLPCC18纠错数据集：[NLPCC18](https://github.com/zhaoyyoo/NLPCC2018_GEC)
+ CTC2021评测比赛：[CTC-2021](https://destwang.github.io/CTC2021-explorer/)

## 联系我们
1. 如果您对数据/代码有任何问题，您可以提交Issue或联系 [`xlxw@zju.edu.cn`](mailto:xlxw@zju.edu.cn)
2. 如果您在使用评测页面有任何问题，您可以联系[`pengjw@zju.edu.cn`](mailto:pengjw@zju.edu.cn)
