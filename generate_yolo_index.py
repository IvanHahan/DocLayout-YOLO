import os

data_dir = "/home/azureuser/cloudfiles/code/datasets/odel_title_value_yolo"
filename = "train.txt"


img_paths = []
for img_file in os.listdir(os.path.join(data_dir, "images")):
    img_paths.append("./images/" + img_file)

with open(os.path.join(data_dir, filename), mode="w") as f:
    f.write("\n".join(img_paths))
