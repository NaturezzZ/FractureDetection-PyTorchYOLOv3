import torch
import torch.nn.functional as F
import numpy as np
from utils.datasets import resize
import os
import pdb
import cv2


def horisontal_flip(images, targets):
    images = torch.flip(images, [2])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def vertical_flip(images, targets):
    images = torch.flip(images, [1])
    targets[:, 3] = 1 - targets[:, 3]
    return images, targets

def crop(images, targets):
    #print(images)
    #print(targets)
    max_row, max_col = targets.shape
    i = np.random.randint(max_row)
    x = targets[i, 2]
    y = targets[i, 3]
    w = targets[i, 4]
    h = targets[i, 5]
    #print((x,y,w,h))
    _, h_factor, w_factor = images.shape
    
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    
    nx1 = np.random.uniform(0, x1)
    nx2 = np.random.uniform(x2, 1)
    ny1 = np.random.uniform(0, y1)
    ny2 = np.random.uniform(y2, 1)
    
    nh = ny2-ny1
    nw = nx2-nx1
    
    x1_p = np.around(w_factor*nx1)
    x2_p = np.around(w_factor*nx2)
    y1_p = np.around(h_factor*ny1)
    y2_p = np.around(h_factor*ny2)
    ty1 = int(y1_p)
    ty2 = int(y2_p)
    tx1 = int(x1_p)
    tx2 = int(x2_p)
    #print((ty1,ty2,tx1,tx2))
    new_image = images[:, ty1:ty2, tx1:tx2]
    #new_image = images[:, :int(y2_p), :]
    #new_image = new_image[:,:,int(x1_p):int(x2_p)]
    new_image = resize(new_image, (1024,1024))
    images = new_image

    result_tar = np.zeros((1,6))
    result_tar[0, 1] = targets[0, 1]
    result_tar[0, 2] = (x-nx1)/nw
    result_tar[0, 3] = (y-ny1)/nh
    result_tar[0, 4] = w/nw
    result_tar[0, 5] = h/nh
    #print(result_tar)
    result_tar.astype(np.double)
    result_tar = torch.from_numpy(result_tar)

    targets = result_tar
    
    return images, targets

def Con_Bright(images, targets):
    """
    调整样本图像的对比度、亮度
    """
    images_np = images.numpy()
    images_np = cv2.convertScaleAbs(images_np * 255, alpha=1.5, beta=-1)
    images_np = images_np / 255
    images = torch.from_numpy(images_np)

    return images, targets


def Sharpen(images, targets):
    """
    卷积锐化图像
    """
    kernel = np.array([
        [[[-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]],
        [[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]],
        [[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        [[-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]]]
    ])
    kernel = kernel.astype(np.float64)
    kernel = torch.from_numpy(kernel)
    images = images.unsqueeze(0)
    images = F.conv2d(images, kernel, stride=1, padding=0)
    images = images.squeeze(0)

    return images, targets
