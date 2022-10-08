from pathlib import Path
import xml.etree.ElementTree as ET

xmldir = r'C:\Users\Admin\Desktop\image 151-200\annotations_test'

xmlnames = sorted(list(Path.iterdir(Path(xmldir))))

bbox_labels_count = {}

for xmlname in xmlnames:
    tree = ET.parse(xmlname)
    root = tree.getroot()

    objects = root.findall('object')

    for node in objects:
        label_node = node.find('name')
        if label_node.text not in bbox_labels_count.keys():
            bbox_labels_count[label_node.text] = 1
        else:
            bbox_labels_count[label_node.text] += 1
            
print("bbox labels count:", bbox_labels_count)