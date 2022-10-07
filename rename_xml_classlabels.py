import xml.etree.ElementTree as ET
import os
from pathlib import Path

xmldir = "./datasets/sp_ppe/VOCdevkit/VOC2012/Annotations"

xmlnames = sorted(list(Path.iterdir(Path(xmldir))))

for xmlname in xmlnames:
    tree = ET.parse(xmlname)
    root = tree.getroot()

    objects = root.findall('object')

    for node in objects:
        label_node = node.find('name')
        if label_node.text == "helmet-mask":
            label_node.text = "helmet_mask"
        elif label_node.text == "helmet-vest":
            label_node.text = "helmet_vest"
        elif label_node.text == "mask-vest":
            label_node.text = "mask_vest"

    tree.write(xmlname)