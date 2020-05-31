from __future__ import print_function

import os
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def show_box(bbox, color):
    # 参数 image 应为 H*W*C 的 numpy 格式
    x1, y1, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    x2, y2 = x1 + width, y1 + height
    plt.plot([x1, x2], [y1, y1], c=color, lw=2) # 颜色，宽度
    plt.plot([x1, x1], [y1, y2], c=color, lw=2)
    plt.plot([x1, x2], [y2, y2], c=color, lw=2)
    plt.plot([x2, x2], [y1, y2], c=color, lw=2)

if __name__ == "__main__":
    img_dir = './data/custom/images/'
    ground_truth_json_dir = './data/json_origin/anno_val.json'
    prediction_json_dir = './data/json_origin/out.json'
    save_dir = './output_withbox_dpi600/'

    with open(ground_truth_json_dir, "r") as f:
        val = json.load(f)
    with open(prediction_json_dir, "r") as f:
        prediction = json.load(f)
    
    images_info = val["images"]
    ground_truth = val["annotations"]
    
    img_cnt = len(images_info)
    for img_i in range(img_cnt):
        print ("dealing with image %d / %d" % (img_i + 1, img_cnt))
        # calculate image position
        img_id = images_info[img_i]["id"]
        img_pos = img_dir + images_info[img_i]["file_name"]
        img_numpy = cv2.imread(img_pos)
        # show image
        plt.figure()
        plt.imshow(img_numpy)
        # load ground truth
        ground_truth_cnt = 0
        for frac_i in range(len(ground_truth)):
            if ground_truth[frac_i]["image_id"] == img_id:
                show_box(ground_truth[frac_i]["bbox"], color = 'y')
                ground_truth_cnt = ground_truth_cnt + 1
        prediction_cnt = 0
        for frac_i in range(len(prediction)):
            if prediction[frac_i]["image_id"] == img_id:
                show_box(prediction[frac_i]["bbox"], color = 'g')
                prediction_cnt = prediction_cnt + 1
        print ("%d.jpg: ground truth cnt %d, prediction cnt %d" % (img_id, ground_truth_cnt, prediction_cnt))
        # save image and print
        plt.axis('off')
        plt.savefig(save_dir + str(img_id) + '_ground_truth_and_prediction.jpg', dpi = 600)
        plt.close()
