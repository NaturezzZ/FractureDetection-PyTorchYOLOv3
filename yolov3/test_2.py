from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import re
import json
from AP50test import AP50_standard_test

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, output_json = False):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, crop_prob = 0)
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
            if output_json:
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
                        
                        row = batch_i * batch_size + frac_i
                        frac_dict["image_id"] = Id[row]
                        frac_dict["category_id"] = 1 # for all fractions in this task category_id = 1
                        x1 = image_fraction_list[i][0]
                        y1 = image_fraction_list[i][1]
                        x2 = image_fraction_list[i][2]
                        y2 = image_fraction_list[i][3]
                        frac_dict["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                        frac_dict["score"] = image_fraction_list[i][4]
                        json_out.append(frac_dict)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    if output_json:
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
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_final.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=12, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    train_path = 'data/custom/train.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute AP on Training Set...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=train_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        output_json=True
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    AP50_standard_test()

    # print("Compute mAP...")

    # precision, recall, AP, f1, ap_class = evaluate(
    #     model,
    #     path=valid_path,
    #     iou_thres=opt.iou_thres,
    #     conf_thres=opt.conf_thres,
    #     nms_thres=opt.nms_thres,
    #     img_size=opt.img_size,
    #     batch_size=8,
    #     output_json=True
    # )

    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    # print(f"mAP: {AP.mean()}")

    # AP50_standard_test()
