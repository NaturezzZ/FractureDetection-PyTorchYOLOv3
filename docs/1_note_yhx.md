输入图像：256*256。

测试了的网络：AlexNet和GoogLeNet，并且都实验了pretrained和untrained的模型。pretrained的模型是已经见过其他什么猫啊狗啊之类的图片的模型。

也就是说，测试的模型有AlexNet-pretrained，AlexNet-untrained，GoogLeNet-pretrained和GoogLeNet-untrained。总体来讲两者不相上下，但是经过数据增强后效果有提升，或者把两个模型混合起来效果也有提升。

参数Epoch=120, learning rate=0.01(untrained)/0.001(pretrained), step down=33%, gama=0.1

数据增强：随机剪裁227*227, mean subtraction, mirror image (文章里说是prebuilt的，那就是说这些数据增强方法应该已经内置在测试的模型里了)

额外的数据增强：旋转,CLAHE(一种特殊的数据增强算法，在论文中用-TA结尾作为标注，如AlexNet-TA，GoogLeNet-TA)

混合两个模型的方法：50%-50%加权，或者其他权重加权

网络的输出应该是二分类？