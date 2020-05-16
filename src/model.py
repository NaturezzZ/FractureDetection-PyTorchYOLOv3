
from config import data_dir, anno_path, output_path, model_path
from config import epochs, learning_rate, batch_size

with open(anno_path, "r") as f:
    data = json.load(f)
print("load file from " + anno_path)


with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
print("write to " + output_path)
