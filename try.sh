#!/bin/bash
source /home/yhx/anaconda3/bin/activate
conda activate pytorch

python train.py --epochs 30 --pretrained_weights change_padding/good_0.0568_ckpt_9.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 30 --pretrained_weights change_padding/good_0.0588_ckpt_3.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 30 --pretrained_weights change_padding/good_0.0588_ckpt_7.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 30 --pretrained_weights change_padding/good_0.0614_ckpt_11.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 30 --pretrained_weights change_padding/good_0.0632_ckpt_13.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 30 --pretrained_weights change_padding/good_0.0833_ckpt_5.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 30 --pretrained_weights checkpoints/good_0.0577_ckpt_1.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001
