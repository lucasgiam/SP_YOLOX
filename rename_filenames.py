import os
from pathlib import Path

imgdir = "./datasets/sp_ppe/VOCdevkit/VOC2012/JPEGImages"
xmldir = "./datasets/sp_ppe/VOCdevkit/VOC2012/Annotations"

def main():
    imgnames = sorted(list(Path.iterdir(Path(imgdir)))) 
    for imgname in imgnames:
        renamerobo(imgname)
        xmlname = Path(xmldir) / imgname.with_suffix(".xml").name
        renamerobo(xmlname)

def append(src):
    src1 = src.stem
    src2, src3 = src1.split("_")
    src4 = src2.replace("+","_")
    src5 = src4 + "_" + src3
    os.rename(src, src.with_stem(src5))
    # print(src)
    # print(src.with_stem(src5))

def renamerobo(src):
    src1 = src.stem
    src2, src3 = src1.split("--")
    src4 = src3.split("-")[0]
    src5 = src2 + "_" + src4
    os.rename(src, src.with_stem(src5))
    # print(src)
    # print(src.with_stem(src5))

if __name__ == "__main__":
    main()