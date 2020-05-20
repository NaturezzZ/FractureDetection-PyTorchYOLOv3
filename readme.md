#### 调研

- 搞清楚evaluation是怎么测的 yhx xjq

- 命令行参数，json解析（+整份代码） yhx

- detectron2 znq lhp yhx https://github.com/facebookresearch/detectron2

- 看看有没有新的论文/能用的代码 lhp

- tensorflow https://github.com/jiangyy5318/medical-rib 

- resnet unet imagenet

- pytorch znq

- 分块辨认？

- 判定之后反向找热力图？

~~V100 超算~~

#### 讨论记录

- 第二次讨论： 2020/5/16 周六晚 21:30

完成内容：znq，lhp：完成detectron2的阅读和基本用法；yhx，xjq：读完evaluation和助教发的reading

讨论内容：evaluation的细节，precision和recall两个评测指标，AP能否反向传播是个问题。detectron2的输入、输出数据格式，网络架构，基本用法。暂时认为只用detectron2得不到较好的效果（不收敛警告）

- 第三次讨论：2020/5/19 周二晚 20:30

完成内容：xjq，lhp：做出来可运行的用detectron2看一下效果；yhx，znq：做rcnn

#### 已完成

/docs/read_write_json  读写json文件的示范。里面有一个readme和一个代码示范

/docs/x_note_xxx   xxx写的第x篇论文读后情况

/docs/data_info.docx     输入数据存储格式

/docs/evaluation_info.md    测试方式（evaluation，AP50）的说明

/src/param.py    处理命令行参数的代码

/src/dataloader.py     处理数据、产生dataloader的代码

/src/model.py     一个朴实无华的CNN



以下代码仍在调试中

/src/train.py    主代码

/src/test.py    接口尚未完工的evaluation代码（暂时替换为直接计算CrossEntropy）



