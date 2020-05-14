Json文件：可以参考b站 可乐编程第26期0基础学python之二十三：JSON数据格式（10分钟）

Json存储类似python的dict和list

输入输出就用python的with open() as

请将测试数据中anno_val.json和这份代码放到同一份目录下，编译

```
python read_write_json.py
```

将会生成一个一模一样的test.json文件。以下是简要说明：

```python
Json文件：可以参考b站 可乐编程第26期0基础学python之二十三：JSON数据格式（10分钟）
Json存储类似python的dict和list
输入输出就用python的with open ... as f

# 使用json库
import json

# 读文件
with open('anno_val.json', "r") as f:
    data = json.load(f)

# 此处data可以直接作为python的数据调用

# 写文件
with open('test.json', "w") as f:
    json.dump(data, f, indent=4)
```

