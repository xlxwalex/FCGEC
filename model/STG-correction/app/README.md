 # FCGEC Reporter
***New feature (2023/04/25):*** Reporter (which can generate detailed report for document-level error correction task.)

新加入的特性：Reporter，可以用于生成文档级别的错误修改报告

---

## 1 Requirements (需要安装的包)
+ ***jieba*** (Optional, the package utilize to convert operate to natural language / jieba包的安装可选，主要用于将操作转化为自然语言)
+ ***docxtpl*** (Important, docxtpl is employed to fill in the docx template / docxtpl包是必须的组件，用于填充docx格式的报告模板)
+ ***libreoffice*** (Optional, the software is used to convert `docx` to `pdf` / 可选的软件，仅用于将docx文件转换为pdf文件)


## 2 Usage(使用方式)

Each parameter in the demo script [`demo_pipeline.py`](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/demo_pipeline.py) has a default value, so it can be run directly:

在上级目录中的DEMO脚本[`demo_pipeline.py`](https://github.com/xlxwalex/FCGEC/blob/main/model/STG-correction/demo_pipeline.py)每个参数都有默认值，因此可以直接运行（其中用于演示的文档文本已放在[`data/demo/`](https://github.com/xlxwalex/FCGEC/tree/main/model/STG-correction/dataset/demo)下）:
```bash
python demo_pipeline.py
```
Meaning of parameters(可选的参数含义)：
+ ***report_title*** :The tile in the report template (输出的报告标题)
+ ***dataset_path*** : Path of document data (需要解析的文档文本文件位置)
+ ***report_prefix*** : Path of output for report data (输出报告的前缀路径，默认为主目录)
+ ***report_type***: The file type of the report file [`docx`, `pdf`] (输出报告的类型，可选`docx`, `pdf`,其中`pdf`需要libreoffice软件)

## 3 Screenshot (报告样例截图)
  <p align="center">
    <br>
    <img src="../../../figure/demo-v2.jpg" width="600"/>
    <br>
  </p>