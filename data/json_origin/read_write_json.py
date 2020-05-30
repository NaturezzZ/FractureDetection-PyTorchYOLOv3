# Json文件：可以参考b站 可乐编程第26期0基础学python之二十三：JSON数据格式（10分钟）

# json 是一种数据格式
# .json 文件可以直接用记事本或者写字板打开
# 可以看到.json里面的内容和python的dictionary和list是一样的
# 我们只要学一下怎么用json自带的包读写.json里的数据就行了
# 读出来以后就编程python里的dictionary和list了

import json # 使用json库

# 读文件
with open('fake.json', "r") as f:
    data = json.load(f)

# 此处data可以直接作为python的数据调用

'''
例如在这个文档中，数据是
{
    "info": {
        "description": "Chest X-Ray Fracture testing Dataset",
        "year": 2020
    },
    "licenses": {},
    "categories": [
    即data是一个dictionary，有data["info"]=..., data["liicenses"]=..., data["categories"]=...
    data["info"]还是一个dict
    data["info"]["description"]是一个字符串"Chest X-Ray Fracture testing Dataset"
'''

# python 正常读取 data 里的内容
# print("description = " + data["info"]["description"])
# print("year = " + str(data["info"]["year"])) # data["info"]["year"] 的类型是 int

# 写文件
# 这里我们把读取到的文档原封不动地输出
with open('out.json', "w") as f:
    json.dump(data, f, indent=4) # 设置缩进为4格