from param import *

with open(anno_path, "r") as f:
    data = json.load(f)
print("load file from " + anno_path)


class FractionDataset():
	def __init__(self, data_dir):
            

with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
print("write to " + output_path)
