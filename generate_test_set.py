import os
import shutil

img_dir = "./datasets/sp_ppe/VOCdevkit/VOC2012/JPEGImages"
test_set_filenames_path = "./datasets/sp_ppe/VOCdevkit/VOC2012/ImageSets/Main/test.txt"
test_set_save_dir = "./datasets_test/sp_ppe_test_set"

images = os.listdir(img_dir)

with open(test_set_filenames_path) as f:
    filenames = f.readlines()
    
for filename in filenames:
    filename = filename.split('\n')[0] + ".jpg"
    for img in images:
        if img == filename:
            shutil.copy(os.path.join(img_dir, img), test_set_save_dir)
