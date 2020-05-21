import cv2
import os
import json
import numpy as np

json_dir = '/home/zhengnq/FractureDetection/data/fracture/annotations/anno_val.json'
img_dir = '/home/zhengnq/FractureDetection/data/fracture/val/'
outimg_dir = './val_outimage/'
outtxt_dir = './val_outtxt/'


with open(json_dir, "r") as f:
    data = json.load(f)

num_of_fractures = len(data['annotations'])
num_of_images = len(data['images'])

for i in range(700):
    image = 0
    if os.access(img_dir+str(i)+'.png', os.R_OK):
        print('pic'+str(i))
        image = cv2.imread(img_dir+str(i)+'.png', os.R_OK)
        print(np.shape(image))
        x_max, y_max = np.shape(image)[1], np.shape(image)[0]
        image = cv2.resize(image,(512,512))
        img = np.zeros((512,512,3))
        img[:,:,0] = image[:,:]
        img[:,:,1] = image[:,:]
        img[:,:,2] = image[:,:]
        print(np.shape(img))
        cv2.imwrite(outimg_dir+str(i)+'.png', img)
        outfile = open(outtxt_dir+str(i)+'.txt', "w")
        for j in range(num_of_fractures):
            image_id = data['annotations'][j]['image_id']
            if str(image_id)==str(i):
                x0, y0, w, h = data['annotations'][j]['bbox'][:]
                outfile.write('0 ' + str((x0+w/2)/x_max) + ' ' + str((y0+h/2)/y_max) + ' ' + str(w/x_max) + ' ' + str(h/y_max)+'\n')
        outfile.close()