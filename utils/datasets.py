import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

from utils.augmentations import horisontal_flip
from utils.augmentations import vertical_flip
from utils.augmentations import crop
from utils.augmentations import Con_Bright
from utils.augmentations import Sharpen
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    # left, right, up, down
    # https://blog.csdn.net/sinat_36618660/article/details/100122745
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    pad = (0, 0, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad




def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=1024):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # https://blog.csdn.net/qq_37385726/article/details/81771980

        # print(img.shape)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize

        img = resize(img, self.img_size)
        
        img, targets = Con_Bright(img, torch.tensor((1,6)))
        
        img, targets = Sharpen(img, targets)
        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=1024, augment=True, multiscale=False, normalized_labels=True, crop_prob=0, final_test = False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.crop_prob = crop_prob
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 64
        self.max_size = self.img_size + 3 * 64
        self.batch_count = 0
        
        if final_test:
            print ("Loading images' size...")
            self.image_w = []
            self.image_h = []
            for index in range(len(self.img_files)):
                img_path = self.img_files[index].rstrip()
                img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
                if len(img.shape) != 3:
                    img = img.unsqueeze(0)
                    img = img.expand((3, img.shape[1:]))
                _, h, w = img.shape
                self.image_w.append(w)
                self.image_h.append(h)
            print ("Load ended.")

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # print(img-transforms.ToTensor()(Image.open(img_path)))
        # Handle images with less than three channels
        


        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            #print('successfully opened '+label_path)
            boxes = np.loadtxt(label_path).reshape(-1, 5)
            boxes.astype(np.double)
            #print(np.loadtxt(label_path).reshape(-1, 5).dtype)
            boxes = torch.from_numpy(boxes)
            # Extract coordinates for unpadded + unscaled image
            #print(boxes.type)
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[0] # warning: changed by yhx; may be wrong!
            y2 += pad[2] # warning: changed by yhx; may be wrong!
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        else:
            print('open failed '+label_path)
        # Apply augmentations

        if np.random.uniform() < self.crop_prob:
            img, targets = crop(img, targets)
        
        img = resize(img, self.img_size)
        
        img, targets = Con_Bright(img, targets)
        
        img, targets = Sharpen(img, targets)

        targets = targets.type(torch.FloatTensor)
        img = img.type(torch.FloatTensor)
        '''
        picture = img.numpy()
        picture= picture*255
        picture = picture.transpose((1,2,0))
        print(np.shape(picture))
        cv2.imwrite(str(index)+'.png', picture)
        '''
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            if np.random.random() < 0.5:
                img, targets = vertical_flip(img, targets)
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
