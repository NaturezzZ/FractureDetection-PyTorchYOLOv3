### 评测相关的包的安装

测试环境：Win10, Visual Studio 2017, anaconda，python 3.7。其余环境请自行做微调。



0) 将数据放置在合适的目录下：

```
./src/test.py
参考答案ground truth放这：
./data/annotations/fracture/instances_val2014.json
你的输出放这：
./data/annotations/fracture/instances_val2014_fakebbox100_results.json
这里用的是2014年coco一个比赛的数据（助教的note_project_4里给的链接），我们放置数据的格式和它是类似的。
```

你也可以打开test.py文档自由修改放置目录。



以下是一些可能遇到的坑

1) 用 pip 安装 pycocotools。

在 anaconda 中输入：

pip install pycocotools

如果遇到报错，见 https://blog.csdn.net/u010103202/article/details/87905029



2) 运行 test.py，查看是否有遇到报错

```
python test.py                (在anaconda终端里输入)
```

这个程序会测试在你的环境下是否能正常评测。如果遇到错误numpy的float错误，请对numpy降级：

```
pip install -U numpy==1.17.0    (在anaconda终端里输入)
```

如果正常运行，那么得到的结果应该和写在test.py最后注释里的结果一致。

