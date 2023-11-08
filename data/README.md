# FCGEC Data Format
Our corpus is stored in `JSON` format, where the format of [`FCGEC_train.json`](https://github.com/xlxwalex/FCGEC/blob/main/data/FCGEC_train.json) and [`FCGEC_valid.json`](https://github.com/xlxwalex/FCGEC/blob/main/data/FCGEC_valid.json) is shown below

我们的语料库是以JSON格式进行存储的，其中[`FCGEC_train.json`](https://github.com/xlxwalex/FCGEC/blob/main/data/FCGEC_train.json) 以及 [`FCGEC_valid.json`](https://github.com/xlxwalex/FCGEC/blob/main/data/FCGEC_valid.json) 的格式如下所示：

## Json Format
```json
{

    "id : The global id of the instance": 
    {
        "sentence": "The original sentence / 原始句子",
        "error_flag": "Whether sentence contains errors / 原句是否是病句",
        "error_type": "The error types of sentence / 原句的错误类型，若为非病句则为*",
        "operation" : [ { "The operation of the first reference / 第一种编辑操作的json字符串" }, 
                        { "The operation of the second reference/ 第二种编辑操作的json字符串" }, 
                        { ... } 
                     ],
         "external" : "Additional information (e.g., version) / 额外信息"
    },
    "id ...":{ 
        {...}
    }
}
```

***Note:*** 
1. Test set [`FCGEC_test.json`](https://github.com/xlxwalex/FCGEC/blob/main/data/FCGEC_test.json) do not have the attributes `error_flag`, `error_type` and `operation`.
2. Due to the presence of homologous sentences among the training, validation, and test sets, which may lead to data contamination, we have provided the `FCGEC_train_filtered.json` file, which excludes sentences in the training set that overlap with those in the validation and test sets.

***注意：*** 
1. 测试集没有给出`error_flag`， `error_type` 以及 `operation`三个属性。
2. 由于训练集、验证集和测试集之间存在同源句子，会导致数据污染。因此我们提供了`FCGEC_train_filtered.json`文件，该文件排除了训练集中与验证集和测试集重叠的句子。

## Operation Format

Suppose the given sentence is "A B C D E".

假设原始句子是 “A B C D E”
### 1. Switch operation（交换操作）
```json
{"Switch":[0,2,1,3,4]}                               // (A B C D E → A C B D E)
```

### 2. Delete operation（删除操作）
```json
{"Delete":[3]}                                       // (A B C D E → A B C E)
```

### 3. Insert operation（插入操作）
```json
{"Insert":[{"pos":1,"tag":"INS_1","label":["F"]}]}  // (A B C D E → A B F C D E)
```

### 4. Modify operation（修改操作）
```json
{"Modify":[{"pos":2,"tag":"MOD_1","label":["F"]}]}  // (A B C D E → A B F D E)
```