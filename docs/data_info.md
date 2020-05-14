数据里前半段是对每个骨折处的描述。

对任意骨折处的注解包括：

"bbox": [x,y,width,height] 有骨折的小矩形的坐标。按我的理解应该x是横过去的（列），y是竖下来的（行），对应width是x往右延伸的宽度，height是y下面垂下来的高度。整张图最左上角的位置是(0,0)不是(1,1)。（如下图中标出的一样）、

![img](file:///C:\Users\A\AppData\Local\Temp\ksohtml17740\wps1.jpg) 

"area": 一定等于width*height

"category_id": 恒为1

"id": 

"iscrowd": 恒为0

"image_id": 对应图像标号（多个骨折可能对应到一张图）

 

数据里后半段是对每个图片的描述。

对任意图片的注解包括：

"id": 647,

"file_name": "647.png",

"height": 3056,

"width": 2544