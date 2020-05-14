用Keras，Tensorflow。

试验了 VGG16/19,Xception/Inception,ResNet。实验的时候直接套用网络的主结构，不改变卷积层结构，但是尝试调整了fc1和fc2（fully connected layers）的大小。

尝试调整了Batch Size,Pool,Dropout。

这些超参数是用一个叫做FGLab的东西自动调整的（我没用过这个，不太了解）

测试了AUC，Sens，Spec，PPV等标准。Sens是不漏诊的概率，Spec是不误诊的概率。