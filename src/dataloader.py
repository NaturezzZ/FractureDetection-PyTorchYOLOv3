"""
（一）本代码中所有传递的信息的格式均为：
{"image": image, "bbox": bbox}
这是某一个 bbox 和它对应的图片（注意多个 bbox 可能对应同一个图片）
1. 其中的 image 是 ndarray (numpy 读取的图片) 的格式，相当于 Height*Weight*Channel 的三维数组。
注意是顺序是 Height*Weight*Channel，也就是 y*x*channel
如果用了 ToTensor 把 image 转换为 Tensor，则会转换为 torch 包的 C*H*W 格式
2. bbox 是 np.array([x, y, weight, height]) 的格式
（二）以下类可以当做函数来使用：
Rescale()  更改大小
Crop()  剪裁（随机/只提取bbox）
ToTensor()  把 ndarray 转换为 Tensor
（三）本文中自己指定的 output_size 都是 tuple 格式，应为 (weight, height)
"""

from __future__ import print_function, division
import os
import cv2
import json
import torch
import numpy as np

from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from param import *

import warnings
warnings.filterwarnings("ignore")

def show_box(image, bbox):
    """
    显示带有黄色 bbox 的图片
    
    参数 image 应为 H*W*C 的 numpy 格式
    
    参数 bbox 应为 np.array([1, 2, 3, 4]) 格式
    """
    plt.figure()
    plt.imshow(image)
    x1, y1, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    print ([x1, y1, width, height])
    x2, y2 = x1 + width, y1 + height
    plt.plot([x1, x2], [y1, y1], c='y', lw=0.5)
    plt.plot([x1, x1], [y1, y2], c='y', lw=0.5)
    plt.plot([x1, x2], [y2, y2], c='y', lw=0.5)
    plt.plot([x2, x2], [y1, y2], c='y', lw=0.5)
    plt.axis('off')
    plt.show()

def check_bbox(new_w, new_h, bbox):
    x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(x1, 0), new_w)
    x2 = min(max(x2, 0), new_w)
    y1 = min(max(y1, 0), new_h)
    y2 = min(max(y2, 0), new_h)
    bbox[0], bbox[1], bbox[2], bbox[3] = x1, y1, x2 - x1, y2 - y1
    if bbox[2] <= 0 or bbox[3] <= 0:
        return np.array([-1, -1, -1, -1])
    return bbox

def calc(sample):
    h = sample['bbox'][2]
    w = sample['bbox'][3]
    s1 = w * h
    h = sample['image'].shape[1]
    w = sample['image'].shape[2]
    s2 = w * h
    return np.array(float(s1) / s2)

class FractionDataset(Dataset):
    """
    把每个骨折转换为可以被 Dataloader 读取的 iterable 格式
    """
    def __init__(self, img_dir, json_dir, pixels = 256, verbose = False, transform = None):
        self.transform = transform

        # load the json file
        with open(json_dir, "r") as f:
            data = json.load(f)
        if verbose:
            print("Successfully load json file from " + json_dir)
        
        self.num_of_fractures = len(data["annotations"])
        self.num_of_images = len(data["images"])

        # load bbox info and image info from json file
        self.bbox = []
        self.belong_to_image_id = []
        for i in range(self.num_of_fractures):
            bbox_tmp = data["annotations"][i]["bbox"]
            for j in range(len(bbox_tmp)):
                bbox_tmp[j] = round(bbox_tmp[j])
            self.bbox.append(bbox_tmp)
            self.belong_to_image_id.append(data["annotations"][i]["image_id"])
                
        self.image_id = []
        self.file_name = []
        self.height = []
        self.width = []
        for i in range(self.num_of_images):
            self.image_id.append(data["images"][i]["id"])
            self.file_name.append(data["images"][i]["file_name"])
            self.height.append(data["images"][i]["height"])
            self.width.append(data["images"][i]["width"])

        # id2index[i] 表示和i.jpg对应是图像序列里的第几张
        self.id2index = []
        for i in range(1000): # no more than 1000 images
            self.id2index.append(i)
        for i in range(self.num_of_images):
            self.id2index[self.image_id[i]] = i
        
        # deal with the images
        self.image = []
        for i in range(self.num_of_images):
            self.image.append(cv2.imread(img_dir + self.file_name[i]))
        if verbose:
            print("Successfully load image file from " + img_dir)
        
        # print the last image to check the correctness
        
        self.cache_sample = []
        for fraction_id in range(self.num_of_fractures):
            imgid = self.belong_to_image_id[fraction_id]
            idx = self.id2index[imgid]
            sample = {'image': self.image[idx], 'bbox': np.array(self.bbox[fraction_id])}
            if self.transform:
                sample = self.transform(sample)
            sample['label'] = torch.from_numpy(calc(sample))
            self.cache_sample.append(sample)
        if verbose:
            print("Successfully cache images ")

    # define len(FractionDataset)
    def __len__(self):
        return self.num_of_fractures

    # define FractionDataset[idx]
    # note: return the ith bbox and its corresponding image
    def __getitem__(self, fraction_id):
        return self.cache_sample[fraction_id]

class Con_Bright(object):
    """
    调整样本图像的对比度、亮度

    Args:
        alpha：对比度系数
        beta：亮度系数

    Args note:
        (1) a=1时是原图；
        (2) a>1时对比度增强，图像看起来更加清晰；
        (3) a<1时对比度减弱，图像看起来变暗；
        (4) b影响图像的亮度，随着增加b (b>0)和减小b (b>0)，图像整体的灰度值上移或者下移, 图像整体变亮或者变暗, 不改变图像的对比度
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        new_image = cv2.convertScaleAbs(image * 255, alpha=self.alpha, beta=self.beta)
        return {'image': new_image, 'bbox': bbox}

class Sharpen(object):
    def __init__(self):
        self.kernel = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        sharpened = cv2.filter2D(image, -1,
                                 self.kernel)
        return {'image': sharpened, 'bbox': bbox}

class Rescale(object):
    """
    将样本中的图像重新缩放到给定大小。
    
    Args:
        output_size (tuple)：所需的输出大小。 =(width, height)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        h, w = image.shape[:2]
        new_w, new_h = self.output_size
        img = transform.resize(image, (new_h, new_w))
        if bbox[0] != -1: # if the box exists
            bbox[0] = round(float(bbox[0]) * new_w / w)
            bbox[1] = round(float(bbox[1]) * new_h / h)
            bbox[2] = round(float(bbox[2]) * new_w / w)
            bbox[3] = round(float(bbox[3]) * new_h / h)
        bbox = check_bbox(new_w, new_h, bbox)
        return {'image': img, 'bbox': bbox}

class Crop(object):
    """
    随机裁剪样本中的图像
    Args:
        output_size (tuple)：所需的输出大小。=(width, height)
       
        type (str): "Random" or "BBox_only"
        if type == BBox_only, the bbox image will be rescaled to output_size
    """

    def __init__(self, output_size, pattern = "Random"):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        assert (pattern == "Random") or (pattern == "BBox_only")
        self.output_size = output_size
        self.pattern = pattern

    # define Crop(sample) so that Crop can be used as a function
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        if self.pattern == "Random":
            # randomly crop the image
            h, w = image.shape[:2]
            new_w, new_h = self.output_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            image = image[top: top + new_h,
                      left: left + new_w]
            if bbox[0] != -1:
                bbox = bbox - [left, top, 0, 0]
            bbox = check_bbox(new_w, new_h, bbox)
            return {'image': image, 'bbox': bbox}
        else:
            # else if type == "BBox_only"
            res = Rescale(self.output_size)
            if bbox[0] == -1:
                return res(sample)
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            image = image[y: y + h, x: x + w, :]
            bbox = np.array([0, 0, w, h])
            result = {'image': image, 'bbox': bbox}
            result = res(result)
            return result

class ToTensor(object):
    """
    将样本中的 ndarrays 转换为 Tensors
    """
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # 交换颜色轴，因为
        # numpy 包的图片是: H * W * C
        # torch 包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbox': torch.from_numpy(bbox)
                }

if __name__ == "__main__":
    img_dir = '../data/fracture/val/'
    json_dir = '../data/fracture/annotations/anno_val.json'

    # test the correctness of FractionDataset class and the transform functions
    print("Start Testing")
    transformed_dataset = FractionDataset(img_dir, json_dir,
                                           verbose = False, # silent, not verbose
                                           transform = transforms.Compose([
                                               Rescale((1024, 1024)),
                                               Crop((512, 512), pattern = "Random"),
                                               Con_Bright(1.5, 0), 
                                               Sharpen(), 
                                               ToTensor()
                                           ]))
    print("Loading ended")

    # 设定调试的时候输出几张图片
    output_number = 5
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['bbox'].size(), sample['label'])
        show_box(sample['image'].numpy().transpose((1, 2, 0)), sample['bbox'].numpy())
        if i == output_number - 1:
            break

'''
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
print("write to " + output_path)
'''