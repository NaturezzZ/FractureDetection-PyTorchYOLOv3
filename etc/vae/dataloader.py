from __future__ import print_function, division
import os
import json
import torch
import numpy as np
from skimage import io, transform  # 用于图像的IO和变换
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
import warnings
warnings.filterwarnings("ignore")


def show_box(image, bbox):
    """
    显示带有黄色 bbox 的图片
    """
    plt.figure()
    plt.imshow(image)
    x1, y1, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    x2, y2 = x1 + width, y1 + height
    plt.plot([x1, x2], [y1, y1], c='y', lw=0.5)
    plt.plot([x1, x1], [y1, y2], c='y', lw=0.5)
    plt.plot([x1, x2], [y2, y2], c='y', lw=0.5)
    plt.plot([x2, x2], [y1, y2], c='y', lw=0.5)
    plt.axis('off')
    plt.show()


class FractionDataset(Dataset):
    """
    把每个骨折转换为可以被 Dataloader 读取的 iterable 格式
    """

    def __init__(self, img_dir, json_dir, pixels=256, verbose=False, transform=None):

        # load the json file
        with open(json_dir, "r") as f:
            data = json.load(f)
        if verbose:
            print("Successfully load json file from " + json_dir)

        num_of_fractures = len(data["annotations"])
        num_of_images = len(data["images"])

        # load bbox info and image info from json file
        self.bbox = []
        self.belong_to_image_id = []
        for i in range(num_of_fractures):
            self.bbox.append(data["annotations"][i]["bbox"])
            self.belong_to_image_id.append(data["annotations"][i]["id"])
        self.image_id = []
        self.file_name = []
        self.height = []
        self.width = []
        for i in range(num_of_images):
            self.image_id.append(data["images"][i]["id"])
            self.file_name.append(data["images"][i]["file_name"])
            self.height.append(data["images"][i]["height"])
            self.width.append(data["images"][i]["width"])

        # deal with the images
        self.image = []
        for i in range(num_of_images):
            self.image.append(io.imread(img_dir + self.file_name[i]))
        if verbose:
            print("Successfully load image file from " + img_dir)

        if verbose:
            print(self.image[0].size)
            show_box(self.image[0], self.bbox[0])

    # define len(FractionDataset)
    def __len__(self):
        return self.num_of_fractures

    # define FractionDataset[idx]
    def __getitem__(self, idx):
        sample = {'image': self.image[idx], 'bbox': self.bbox[i]}
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
        new_h, new_w = self.output_size[0],self.output_size[1]
        
        
        img = cv2.resize(image, (new_h, new_w))
        #img = transform.resize(image, (new_h, new_w))
        #print(np.shape(img))
        #bbox = bbox * [new_w / w, new_h / h, new_w / w, new_h / h]
        bbox[0] = round(bbox[0]/w)
        bbox[1] = round(bbox[1]/h)
        bbox[2] = round(bbox[2]/w)
        bbox[3] = round(bbox[3]/h)
        return {'image': img, 'bbox': bbox}


class Crop(object):
    """
    随机裁剪样本中的图像

    Args:
        output_size (tuple)：所需的输出大小。

        type (str): "Random" or "BBox_only"

        if type == BBox_only, the bbox image will be rescaled to output_size
    """

    def __init__(self, output_size, pattern="Random"):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        assert (pattern == "Random") or (pattern == "BBox_only")
        self.output_size = output_size
        self.type = pattern

    # define Crop(sample) so that Crop can be used as a function
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        if self.type == "Random":
            # randomly crop the image
            h, w = image.shape[:2]
            new_h, new_w = self.output_size
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            image = image[top: top + new_h,
                          left: left + new_w]
            bbox = bbox - [left, top, left, top]
            return {'image': image, 'bbox': bbox}
        else:
            # else if type == "BBox_only"
            x, y, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
            x = round(x)
            y = round(y)
            h = round(h)
            w = round(w)
            #print((x,y,h,w))
            image = image[x: x + w, y: y + h,:]
            res = Rescale(self.output_size)
            image = res({'image':image, 'bbox':[0, 0, self.output_size[0], self.output_size[1]]})
            
            return image
            #return {'image': image, 'bbox': [0, 0, self.output_size[0], self.output_size[1]] }


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
    data = FractionDataset(img_dir, json_dir, verbose=True)
    '''
    transformed_data = FaceLandmarksDataset(img_dir, json_dir,
                                           verbose = True,
                                           transform = transforms.Compose([
                                               Rescale(256),
                                               Crop(128),
                                               ToTensor()
                                           ]))
    '''
'''
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
print("write to " + output_path)
'''
