import argparse
import os
import torch

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--anno_path", default="", type=str)
parser.add_argument("--output_path", default="", type=str)
parser.add_argument("--model_path", default="models/", type=str)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--batch_size", default=32, type=int)

arg = parser.parse_args()
if arg.device == "cpu":
	device = torch.device("cpu")
else:
	device = torch.device("cuda:0")

data_dir = arg.data_dir
anno_path = arg.anno_path
output_path = arg.output_path
model_path = arg.model_path
epochs = arg.epochs
learning_rate = arg.learning_rate
batch_size = arg.batch_size
