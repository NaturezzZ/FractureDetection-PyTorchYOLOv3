import torch
import torch.nn.functional as F
import numpy as np
from utils.datasets import resize

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def crop(images, targets):
    x = targets[:, 2]
    y = targets[:, 3]
    w = targets[:, 4]
    h = targets[:, 5]
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
    
    new_image = images[:, ty1:ty2, tx1:tx2]
    #new_image = images[:, :int(y2_p), :]
    #new_image = new_image[:,:,int(x1_p):int(x2_p)]
    new_image = resize(new_image, (512,512))
    images = new_image

    targets[:, 2] = (x-nx1)/nw
    targets[:, 3] = (y-ny1)/nh
    targets[:, 4] = w/nw
    targets[:, 5] = h/nh

    return images, targets
