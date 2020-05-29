from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import re
import sys
import time
import datetime
import argparse
import tqdm
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from AP50test import AP50_standard_test


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, final_test = False):
    model.eval()

    # Get dataloader
    if final_test:
        print (path)
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, crop_prob = 0, final_test=final_test)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    json_out = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        #print(batch_i, (_, imgs, targets))
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # print("***************")
            # print(outputs)
            if final_test:
                # open txt file to get image_id for each image
                with open(path, "r") as file:
                    img_files = file.readlines()
                Id = []
                for i in range(len(img_files)):
                    Id.append(int(re.findall(r'\d+', img_files[i])[0]))
                # print (Id)

                for frac_i in range(len(outputs)):
                    if outputs[frac_i] == None:
                        continue
                    sz = outputs[frac_i].size(0)
                    image_fraction_list = outputs[frac_i].numpy().tolist()
                    # print("******************")
                    # print(image_fraction_list)
                    for i in range(sz):

                        frac_dict = {}
                        
                        index = batch_i * batch_size + frac_i
                        frac_dict["image_id"] = Id[index]
                        frac_dict["category_id"] = 1 # for all fractions in this task category_id = 1

                        x_rate = dataset.image_w[index] / img_size
                        y_rate = dataset.image_h[index] / img_size

                        x1 = image_fraction_list[i][0] * x_rate
                        y1 = image_fraction_list[i][1] * y_rate
                        x2 = image_fraction_list[i][2] * x_rate
                        y2 = image_fraction_list[i][3] * y_rate
                        
                        frac_dict["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                        frac_dict["score"] = image_fraction_list[i][4]
                        json_out.append(frac_dict)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    if final_test:
        with open('./data/json_origin/out.json', "w") as f:
            json.dump(json_out, f, indent=4)
    if len(sample_metrics) == 0:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0], dtype=np.int)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=14, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1024, help="size of each image dimension")
    
    parser.add_argument("--data_dir", type=str, default="data/custom/images", help="path to image directory")
    parser.add_argument("--anno_path", type=str, default="data/json_origin/anno_val.json", help="path to thr ground truth ans.json")
    parser.add_argument("--output_path", type=str, default="data/json_origin/out.json", help="path to model output out.json")
    parser.add_argument("--model_path", type=str, default="checkpoints/yolov3_ckpt_final.pth", help="path to mode (weights file)")
    
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Open json file and count the number of images
    with open(opt.anno_path, "r") as f:
        json_data = json.load(f)
    img_num = len(json_data["images"])
    frac_num = len(json_data["annotations"])
    
    # locate valid_path (default /data/custom/valid.txt)
    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]

    # prepare /data/custom/valid.txt
    with open(valid_path, "w") as f:
        for img_i in range(img_num):
            f.write(opt.data_dir + '/' + json_data["images"][img_i]["file_name"] + '\n')
    
    # prepare /data/custom/labels/i.txt
    # open new file /data/custom/labels/i.txt
    Real_Id = {}
    for img_i in range(img_num):
        img_id = json_data["images"][img_i]["id"]
        Real_Id[str(img_id)] = img_i
        with open(opt.data_dir + '/' + str(img_id) + ".txt", "w") as f:
            pass
    # iterate over each fraction and append corresponding /data/custom/labels/i.txt
    for frac_i in range(frac_num):
        # now: the ith annotation
        now = json_data["annotations"][frac_i]
        # load x, y and img_w, img_h
        x1, y1, w, h = now["bbox"][0], now["bbox"][1], now["bbox"][2], now["bbox"][3]
        img_id = now["image_id"]
        img_real_id = Real_Id[str(img_id)]
        img_w = json_data["images"][img_real_id]["width"]
        img_h = json_data["images"][img_real_id]["height"]
        # calculate xmid, ymid and normalize xmid, ymid, w, h to [0, 1]
        x2, y2 = x1 + w, y1 + h
        xmid, ymid = (x1 + x2) / 2, (y1 + y2) / 2
        xmid /= img_w
        ymid /= img_h
        w /= img_w
        h /= img_h
        # append data
        with open(opt.data_dir + '/' + str(img_id) + ".txt", "a") as f:
            s = str(0) + ' ' + str(xmid) + ' ' + str(ymid) + ' ' + str(w) + ' ' + str(h)
            f.write(s)

    # load classes_names
    class_names = load_classes(data_config["names"])
    
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.model_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.model_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.model_path))
    
    print("Compute mAP...")
    # evaluate the model with final_test = True
    # which means that a .json file will be created
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        final_test=True
    )
    
    # AP (provided by YOLOV3 default)
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    
    # AP (provided by the project)
    AP50_standard_test(opt.anno_path, opt.output_path, img_num)
