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

        if verbose:
            for fraction_id in range(3):
            # fraction_id = num_of_fractures - 1
                imgid = self.belong_to_image_id[fraction_id]
                idx = self.id2index[imgid]
                print(idx)
                show_box(self.image[idx], self.bbox[fraction_id])

    # define len(FractionDataset)
    def __len__(self):
        return self.num_of_fractures

    # define FractionDataset[idx]
    # note: return the ith bbox and its corresponding image
    def __getitem__(self, fraction_id):
        imgid = self.belong_to_image_id[fraction_id]
        idx = self.id2index[imgid]
        sample = {'image': self.image[idx], 'bbox': np.array(self.bbox[fraction_id])}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    """
    将样本中的图像重新缩放到给定大小。
    
    Args:
        output_size (tuple)：所需的输出大小。
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
        new_h, new_w = self.output_size
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
        output_size (tuple)：所需的输出大小。
       
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
            image = image[x: x + w, y: y + h]
            result = {'image': image, 'bbox': np.array([0, 0, w, h])}
            result = res(result)
            return result

class ToTensor(object):
    """ 将样本中的 ndarrays 转换为 Tensors"""
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # 交换颜色轴，因为
        # numpy 包的图片是: H * W * C
        # torch 包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbox': torch.from_numpy(bbox)}

if __name__ == "__main__":
    img_dir = '../data/fracture/val/'
    json_dir = '../data/fracture/annotations/anno_val.json'
    
    # test the correctness of FractionDataset class
    print("Start Teststing 1")
    data = FractionDataset(img_dir, json_dir, verbose = True) # verbose

    # test the correctness of transform
    print("Start Teststing 2")
    transformed_dataset = FractionDataset(img_dir, json_dir,
                                           verbose = False, # silent, not verbose
                                           transform = transforms.Compose([
                                               Rescale((512, 512)),
                                               Crop((400, 400), pattern = "BBox_only"),
                                               ToTensor()
                                           ]))
    
    print("Loading ended")
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['bbox'].size())
        show_box(sample['image'].numpy().transpose((1, 2, 0)), sample['bbox'].numpy())
        if i == 2:
            break

'''
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
print("write to " + output_path)
'''