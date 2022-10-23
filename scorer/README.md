# FCGEC Correction Scorer

---

## Metric for Classification Task
We regard the error detection task and error type identification task as classification tasks. Therefore, we adopt four common metrics, i.e., `Accuracy`, `Precision`, `Recall` and `Macro F1 score` to evaluate the model performance.

我们将`错误检测任务`以及`类型判定任务`作为两个分类任务，因此可以使用常用的四个指标：`Accuracy`, `Precision`, `Recall` and `Macro F1 score` 作为衡量模型性能的依据

## Metric for Correction Task
We employ two different metrics : 
+ ***Exact Match metric:*** it is obtained by calculating the percentage of corrected sentences for model outputs that exactly matched with the golden references. 
+ ***The character-level edit metric:*** it is proposed by [MuCGEC](https://github.com/HillZhang1999/MuCGEC) are utilized to compute fine-grained model performance. After obtaining the optimal sequence for character-level editing, they merge consecutive edits of the same type into span-level for both model outputs and golden references. Then MuCGEC calculates the highest `Precision`, `Recall` and `F0.5 score` by comparing the edits of model outputs with each golden reference.

***Note***: You can access MUCGEC repository through the `ChERRANT @ dac5c3f` above. The calculation program is in the `scorers/ChERRANT` path from their repository. And all the punctuations need to be removed before calculating the metrics.

对于`纠错任务`，我们采用了两个指标：
+ ***完全匹配***: 模型纠错输出正确的样本（模型的纠正输出在候选答案中）占全部错误样本的百分比
+ ***字符级编辑指标***：这是[MuCGEC](https://github.com/HillZhang1999/MuCGEC)中提出的衡量纠错性能的指标，更详情可以参考他们的论文。

***注意***：你可以通过上面的`ChERRANT @ dac5c3f`文件夹进入MuCGEC作者的仓库，计算程序在它们仓库的`scorers/ChERRANT`路径下。需要注意的是，在计算以上两个指标前，请去掉文本中所有的标点符号。

## Online Evaluation
We have established an evaluation page in codalab for everyone to evaluate their systems on test set. The link url is: [https://codalab.lisn.upsaclay.fr/competitions/8020](https://codalab.lisn.upsaclay.fr/competitions/8020). If you have an question when using evaluation system, feel free to contact [pengjw@zju.edu.cn](pengjw@zju.edu.cn).

我们在Codalab上建立了一个公共的评测页面，任何人都可以通过该页面评估他们模型在测试集上的性能。评测页面的链接是： [https://codalab.lisn.upsaclay.fr/competitions/8020](https://codalab.lisn.upsaclay.fr/competitions/8020) ，如果您在使用评测页面时遇到了任何问题，请联系 [pengjw@zju.edu.cn](pengjw@zju.edu.cn).