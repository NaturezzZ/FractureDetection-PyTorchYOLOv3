#!/bin/bash
source /home/yhx/anaconda3/bin/activate
conda activate pytorch
python train.py --epochs 250 --evaluation_interval 10 --batch_size 2 --gradient_accumulations 16 --pretrained_weights checkpoints/no_crop.pth --checkpoints_dir crop --crop_probility 1

python train.py --epochs 250 --evaluation_interval 10 --batch_size 2 --gradient_accumulations 16 --pretrained_weights checkpoints/no_crop.pth --checkpoints_dir checkpoints --crop_probility 0 --lr 0.0001
