# FractureDetection

A Yolo-v3 based rib fracture detection model by PyTorch.

Term project of Machine Learning, a Peking University course.

## Introduction

Medical imaging is an essential technique widely used for medical diagnostics. However, analyzing radiographs and detecting fractures require great manual work. Hence, our project is intended to automatically do rib fracture detection, trying to circle out fractures in radiographs with high confidence.

We trained our model on Ubuntu Server 16.04 LTS with NVIDIA RTX2070 Super 8G GPU, AMD 3700X and 32G Memory.  

We used part of the code from [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and open sourced our project based on [GNU General Public License v3.0](https://raw.githubusercontent.com/NaturezzZ/FractureDetection-PyTorchYOLOv3/master/LICENSE).

## Installation

```
$ git clone https://github.com/NaturezzZ/FractureDetection-PyTorchYOLOv3.git
$ cd FractureDetection/
$ sudo pip3 install -r requirements.txt
```

##### Model Downloading

You can download our weights from [Peking University Cloud](https://disk.pku.edu.cn:443/link/4E0333B716CB2D7124ECFFA3624BEF3E) or [OneDrive Cloud](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhengnaiqian_pku_edu_cn/EQPrYihpdNJOncFh7itykQYBw8knjsXMiyDT6t_NO5_DXg?e=hMaa8d), or you can download this file [directly](http://pub1806cloud.zhengnq.top:18060/index.php/s/jSNDWbPd2a4F5pL).

##### Requirements

```
numpy==1.17
torch>=1.0
torchvision
matplotlib
tensorflow
tensorboard
terminaltables
pillow
tqdm
opencv-python
cython
pycocotools
scipy
scikit_image
```

## Usage

##### Test the performance

```
python test.py --data_dir /path/to/fracture/test \
               --anno_path /path/to/anno_test.json \
               --output_path /path/to/output/results.json \
               --model_path /path/to/model/file
```

**Notes:**

1. Input json file (anno_path) **MUST** have the same format as **COCO**, and output json file is also in the  same format.

2. Our evaluation is computed with `AP50` metrics.

3. With default setting of batch size (batch_size=2), test.py consumes no more than 4GB GPU Memory, and you can set batch_size=1 when GPU memory is insufficient.

4. You can change other parameters.

|arguments|usage|
|:---:|:---:|
|data_dir|path to fracture data|
|anno_path|path to COCO ground truth|
|output_path|path to output file|
|model_path|path to model .pth file|
|conf_thres|object confidence threshold|

##### Train the network

You should use `data/data_preparation/change_data.py` to convert data from COCO format to yolov3 format. Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled [0, 1], and the label_idx should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

You should create your own config/yolov3-custom.cfg file with config/create_custom_model.sh, and compute anchor boxes for yolo layer with compute_anchorbox.py.

Finally, you are able to train on custom datasets. 

```
python train.py --pretrained_weights /path/to/pretrained/weights \
                --epochs 100 \
                --data_config config/custom.data
```

You can refer to [PyTorch-YOLOv3](https://github.com/NaturezzZ/FractureDetection-PyTorchYOLOv3/blob/master/yolo_README.md) for more information.

## Maintainers

[@NaturezzZ](https://github.com/NaturezzZ) | [@yanghaoxiang7](https://github.com/yanghaoxiang7) | [@jiaqi-xi](https://github.com/jiaqi-xi) | [@phoenixrain-pku](https://github.com/phoenixrain-pku)

All rights reserved.
