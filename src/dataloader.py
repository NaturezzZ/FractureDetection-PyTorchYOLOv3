from param import *
from torch.utils.data import Dataset
import cv2

train_img_dir = arg.train_img_dir
test_img_dir = arg.test_img_dir
train_json_dir = arg.train_json_dir
test_json_dir = arg.test_json_dir

class FractionDataset(Dataset):
	def __init__(self, img_dir, json_dir, pixels = 256, verbose = False):
        
        # load the json file
        with open(json_dir, "r") as f:
            data = json.load(f)
        if (verbose):
            print("load file from " + json_dir)
        
        num_of_fractures = len(data["annotations"])
        num_of_images = len(data["images"])
        
        # load bbox info and image info from json file
        for i in range(num_of_fractures):
            bbox[i] = data["annotations"][i]["bbox"]
            belong_to_image_id[i] = data["annotations"][i]["id"]
        for i in range(num_of_images):
            image_id[i] = data["images"][i]["id"]
            file_name[i] = data["images"][i]["file_name"]
            height[i] = data["images"][i]["height"]
            width[i] = data["images"][i]["width"]
        
        # deal with the images
        for i in range(num_of_images):
            image[i] = cv2.imread(file_name[i])
        


if __name__ == "__main__":
    img_dir = '../data/fracture/val/'
    json_dir = '../data/fracture/annotations/anno_val.json'
    data = FractionDataset(img_dir, json_dir, verbose = True)

'''
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
print("write to " + output_path)
'''