#!/bin/bash
source /home/yhx/anaconda3/bin/activate
conda activate pytorch

python train.py --epochs 20 --pretrained_weights /mnt/F/0530checkpoints_finetuning/good_0.11118_ckpt_16.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights /mnt/F/0530checkpoints_finetuning/good_0.10588_ckpt_19.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights /mnt/F/0530checkpoints_finetuning/good_0.10227_ckpt_7.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights /mnt/F/0530checkpoints_finetuning/good_0.10102_ckpt_3.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights /mnt/F/0530checkpoints_finetuning/good_0.11316_ckpt_2.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights /mnt/F/0529checkpoints_finetuning/good_0.0892_ckpt_1.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights /mnt/F/0529checkpoints_finetuning/good_0.0878_ckpt_5.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights change_padding/good_0.0588_ckpt_7.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights change_padding/good_0.0833_ckpt_5.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001

python train.py --epochs 20 --pretrained_weights change_padding/good_0.0588_ckpt_3.pth --checkpoints_dir /mnt/F/0530checkpoints_finetuning --evaluation_interval 1 --lr 0.0001