#### 最新消息

dataloader 已经基本完工。

test data比较小，加载不到10s，完全加载的话内存大概为2G不到，可以在自己电脑上测试。

training data约为test data的10倍大小，加载大概1min多一点，完全加载的整个过程内存最大开销大概为16G，建议在服务器上测试

（我也不知道为什么1G的图片加载进内存就变成16G了，不过问题也不大就是了都能放得下）

这份代码的数据都是预处理好的（包括一直到转换为tensor的步骤，从FractionDataSet读取出来直接就是tensor了），读取是比较快的（不需要边读取边transform）

#### 已完成

/docs/read_write_json 读写json文件的示范。里面有一个readme和一个代码示范

/docs/x_note_xxx xxx写的第x篇论文读后情况

/docs/data_info.docx 输入数据存储格式

/docs/evaluation_info.md 测试方式（evaluation，AP50）的说明

/docs/evaluation_metrics_definition.md 测试方式的数学定义说明

/docs/Detectron2_Usage.md Detectron2在json格式的COCO数据集上的使用说明，以及detectron2相关资源的网址

/src/param.py 处理命令行参数的代码

/src/dataloader.py 处理数据、产生dataloader的代码

/src/model.py 一个朴实无华的CNN

/src/d2trial.py, /src/d2test.py detectron2方法，GPU问题尚待解决

以下代码仍在调试中

/src/train.py 主代码

/src/test.py 接口尚未完工的evaluation代码（暂时替换为直接计算CrossEntropy）

#### 调研

-   detectron2 znq lhp yhx https://github.com/facebookresearch/detectron2
-   看看有没有新的论文/能用的代码 lhp
-   tensorflow https://github.com/jiangyy5318/medical-rib
-   resnet unet imagenet
-   分块辨认？
-   判定之后反向找热力图？

#### 讨论记录

-   第二次讨论： 2020/5/16 周六晚 21:30

完成内容：znq，lhp：完成detectron2的阅读和基本用法；yhx，xjq：读完evaluation和助教发的reading

讨论内容：evaluation的细节，precision和recall两个评测指标，AP能否反向传播是个问题。detectron2的输入、输出数据格式，网络架构，基本用法。暂时认为只用detectron2得不到较好的效果（不收敛警告）

-   第三次讨论：2020/5/20 周三晚 21:00

完成内容：xjq，lhp：做出来可运行的用detectron2看一下效果；yhx，znq：做rcnn

-   第四次讨论：2020/5/23 周六下午 3:00

yhx: evaluation api, dataloader change

znq, lhp: yolo api

xjq: detectron2 api

讨论结果：继续使用znq的算法进行进一步工作。

任务：xjq继续对数据进行预处理 znq yhx研究AP、做实验 lhp研究yolo v3

-   第五次讨论：2020/5/25 周一晚 20:30