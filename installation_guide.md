## 0. Prerequisities

You should have the following installed on your workstation:
* Anaconda
* NVIDIA CUDA toolkit and cuDNN driver


## 1. Installation

* Launch the terminal window, create a virtual environment and activate it.

```
conda create -n YOLOX
conda activate YOLOX
```

* Install all the required packages below:

```
pip install -r requirements.txt
pip install -v -e .
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U peekingduck
```


## 2. Load your own custom dataset

* This section assumes that you have already collected your image data and annnotated the images to the Pascal VOC format.

* Open ```YOLOX/datasets``` folder and create the following directory. Note that <dataset_name> is the name of the dataset, for example ```sp_ppe_voc_all_combinations```.

📁 YOLOX/
📄 ├── datasets/
📄 │   ├── VOCdevkit/
📄 │   │   ├── VOC2012/
📄 │   │   │   ├── <dataset_name>/
📄 │   │   │   │   ├── Annotations/
📄 │   │   │   │   ├── ImageSets/
📄 |   |   |   |   |   ├── Main/
📄 |   |   |   |   |   |   ├── train.txt
📄 |   |   |   |   |   |   ├── val.txt
📄 |   |   |   |   |   |   ├── test.txt
📄 │   │   │   │   ├── JPEGImages/

* Transfer all the images (.jpg) into the ```JPEGImages``` folder and all the annotations (.xml) into the ```Annotations``` folder.

* Go into ```train_val_test_split.py``` in the root directory and configure the root_path, seed, train_percent and val_percent, and then run the script using the command:

```python train_val_test_split.py```

* Next, go to ```./yolox/data/datasets```, open the two files ```coco_classes.py``` and ```voc_classes.py```, and set the class labels accordingly.

* See example below for ```sp_ppe_1``` dataset:

VOC_CLASSES = (
    "no_ppe",
    "all_ppe",
    "helmet",
    "mask",
    "vest",
    "mask_vest",
    "helmet_mask",
    "helmet_vest",
)


## 3. Download pre-trained weights

* In the root directory, enter the following command to download pre-trained weights into the ```weights``` folder. Note that you can change ```yolo_m``` to ```yolo_s``` or ```yolo_l```, depending on the size of the model that you want.

``` wget.exe -P weights/pretrained https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth ```

(Note: If you do not already have ```wget.exe``` installed on your computer, you can refer to this video on how to do so: https://youtu.be/CkpTEJH6xkg)


## 4. Train the model

* Create your own ```exp``` file in the "./exps/custom" folder and configure all the necessary training parameters in this ```exp``` file. You can reference the code of the ```exp``` file from ```./exps/custom/sp_ppe_1.py``` or ```./exps/example/yolox_voc/yolox_voc_s.py```.

* Run the following command to initiate model training:

```
python tools/train.py -expn <exp_name> -f <path_to_exp_file> -d <num_gpus> -b <batch_size> --fp16 -o -c <path_to_pretrained_weights>
```

* For example:

```
python tools/train.py -expn sp_ppe -f .\exps\custom\sp_ppe.py -d 1 -b 16 --fp16 -o -c .\weights\pretrained\yolox_s.pth
```

* Once training is completed, go to the output directory where the weights are saved to and copy the ```best_ckpt.pth``` file to the ```weights``` folder, under a <exp_name> subfolder.


## 5. Package the model into PeekingDuck custom node

* Refer to https://peekingduck.readthedocs.io/en/latest/tutorials/03_custom_nodes.html on how to create PeekingDuck custom nodes (if you are not already familiar with it).

* To package the model into PeekingDuck custom node, we will need to create and configure two files, ```<exp_name>.py``` (source code) and ```<exp_name>.yaml``` (config file), in the standard directory format as mentioned in the PeekingDuck custom node tutorial. You can reference the code of these two files from ```./src/custom_nodes/model/sp_ppe_1_yolo_m.py``` (source code) and ```./src/custom_nodes/configs/model/sp_ppe_1_yolo_m.yml``` (config file).


## 6. Run the model

* Open the ```pipeline_config.yml``` file in the YOLOX root directory and put in order all the necessary nodes to form a pipeline. This includes specifying the input source, custom nodes for model inference and bboxes post-processing, as well as the output directory of media and csv logs.

* In the terminal window, enter the following command:

```peekingduck run```