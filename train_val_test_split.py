import os
import random
import numpy as np


# User configs
root_path = "./datasets/sp_ppe/VOCdevkit/VOC2012"
seed = 2022
train_percent = 0.8
val_percent = 0.2


# Calculate test_percent from train_percent and val_percent
test_percent = 1 - train_percent - val_percent


# Define file path for images and annotations
xmlfilepath = root_path + "/Annotations/"
if not os.path.exists(xmlfilepath):
    os.makedirs(xmlfilepath)

imagefilepath = root_path + "/JPEGImages/"
if not os.path.exists(imagefilepath):
    os.makedirs(imagefilepath)

txtsavepath = root_path + "/ImageSets/Main"
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)


# Initialise random seed
print("seed: ", seed)
random.seed(seed)
np.random.seed(seed)


# Assign indices for train-val-test split
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
indices = list(range(num))
np.random.shuffle(indices)
train_num = round(num * train_percent)
val_num = round(num * val_percent)
test_num = round(num * test_percent)
train = indices[:train_num]
val = indices[train_num:train_num+val_num]
test = indices[train_num+val_num:]
print("train size: ", len(train))
print("val size: ", len(val))
print("test size: ", len(test))


# Write image names to train, val and test scripts
ftest = open(txtsavepath + "/test.txt", "w")
ftrain = open(txtsavepath + "/train.txt", "w")
fval = open(txtsavepath + "/val.txt", "w")

for i in indices:
    name = total_xml[i][:-4] + "\n"
    if i in train:
        ftrain.write(name)
    elif i in val:
        fval.write(name)
    else:
        ftest.write(name)

ftrain.close()
fval.close()
ftest.close()