# FCGEC Scripts
Here are some useful tool scripts.

---

## 1 Operates2Seq
***(2022/12/06) [convert_fcgec_to_seq2seq.py](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/scripts/convert_fcgec_to_seq2seq.py)***

This script is utilized to convert the operation labels in our FCGEC data into text sequences to facilitate training for Seq2Seq models.

该脚本用于将FCGEC的操作标签转化为文本序列，方便直接用于Seq2Seq模型的训练

***Usage(使用方式)：***

Each parameter in the script has a default value, so it can be run directly:

每个参数都有默认值，因此可以直接运行:
```bash
python convert_fcgec_to_seq2seq.py
```
Meaning of parameters(可选的参数含义)：
+ ***out_uuid*** : Whether to output uuid column [True / False] (是否输出uuid列)
+ ***data_dir*** : Path of FCGEC data (FCGEC数据文件夹)
+ ***out_dir*** : Path of output for seq2seq data (输出文件夹，默认和FCGEC数据文件夹相同)
+ ***train_file***: Name of train data file (训练集的文件名，为空则不处理训练集)
+ ***valid_file***: Name of valid data file (验证集的文件名，为空则不处理验证集)
+ ***test_file***: Name of test data file (测试集的文件名，为空则不处理测试集)
+ ***out_errflag***: Whether to output out_errflag column [True / False] (是否输出`error_flag`，用于指示是否为病句)
+ ***out_errtype***: Whether to output out_errtype column [True / False] (是否输出`error_type`，用于指定病句的错误类型)

***Note(注意)：*** Since we have multiple references for our operation labels, the output will also have multiple sequences. We use `\t` as a separator between them. 

我们的操作标签有多个参考，因此输出也会有多个矫正文本的序列，它们之间是用`\t`作为分隔的。

## 2 Seq2Operate
***(2023/01/25) [convert_seq2seq_to_operation.py](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/scripts/convert_seq2seq_to_operation.py)***

This script is utilized to convert the seq2seq data to operation labels as our FCGEC data into facilitate training for STG models.
该脚本用于将Seq2Seq的数据转化为FCGEC的操作标签类型数据（该脚本实现为论文的Algorithm 1），方便直接用于STG模型的训练

***Usage(使用方式)：***

We provide a kernel method - min_dist_opt(sentence1, sentence2) for converting single instance, while `sentence1` is the original sentence and `sentence2` is the corrected sentence.

我们在脚本中提供了一个核心方法 - min_dist_opt(sentence1, sentence2)来转换单个的实例（错误-修正的句子对），其中`sentence1`是原始句子(带有语病的句子)而`sentence2`则是修改后的正确句子

Examples for the four types of operation (Switch, Delete, Insert and Modify) are demonstrated in the py script.

我们在python脚本中展示了四种操作方法(Switch, Delete, Insert 和 Modify)的例子

Meaning of parameters(可选的参数含义)：
+ ***COLLOCATION*** : (Flag in Line 16) Whether for adapting to collocation searching for Modify operations (是否为Modify操作开启搭配查找模式)

***Example for COLLOCATION flag (COLLOCATION开关的例子):***

Sentence: 经典计算机无法解决的大规模计算难题提取有效解决方案。

Correction: (提取->提供)经典计算机无法解决的大规模计算难题提供有效解决方案。

* `COLLOCATION` = `TRUE` : Operation = {'Modify': [{'pos': 47, 'tag': 'MOD_2', 'label': '提供'}]}
* `COLLOCATION` = `FALSE` : Operation = {'Modify': [{'pos': 48, 'tag': 'MOD_1', 'label': '供'}]}

***Note(注意)：*** `COLLOCATION` mode rely on the `jieba` package (`COLLOCATION`需要先安装`jieba`库).
