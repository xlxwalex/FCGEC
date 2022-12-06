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
Meaning of parameters(参数含义)：
+ ***out_uuid*** : Whether to output uuid column [True / False] (是否输出uuid列)
+ ***data_dir*** : Path of FCGEC data (FCGEC数据文件夹)
+ ***out_dir*** : Path of output for seq2seq data [True / False] (输出文件夹，默认和FCGEC数据文件夹相同)
+ ***train_file***: Name of train data file (训练集的文件名，为空则不处理训练集)
+ ***valid_file***: Name of valid data file (验证集的文件名，为空则不处理验证集)
+ ***out_errflag***: Name of test data file (是否输出`error_flag`，用于指示是否为病句)
+ ***out_errtype***: Name of test data file (是否输出`error_type`，用于指定病句的错误类型)

***Note(注意)：*** Since we have multiple references for our operation labels, the output will also have multiple sequences. We use `\t` as a separator between them. 

我们的操作标签有多个参考，因此输出也会有多个矫正文本的序列，它们之间是用`\t`作为分隔的。