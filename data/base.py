import os
from typing import Any, List

from PIL import Image
import xml.etree.ElementTree as ET

from torch.utils import data

"""
适用于VOC格式数据集
train.txt 保存训练集数据名
val.txt 保存测试集数据名
"""

def listNames(root: str, flag = "train") -> List[str]:
    _namePath = r"ImageSets\Segmentation\train.txt" if flag == "train" else r"ImageSets\Segmentation\val.txt"
    namePath = os.path.join(root, _namePath)
    names = []
    with open(namePath, "r", encoding="utf-8") as f:
            names = f.read().strip().split('\n')
    return names

class SegDataset(data.Dataset):
    def __init__(self, root: str, flag = "train") -> None:
        self.imgs = []
        self.gts = []
        imgRoot = os.path.join(root, r"JPEGImages")
        gtRoot = os.path.join(root, r"SegmentationClass")
        names = listNames(root, flag)
        for name in names:
            imgPath = os.path.join(imgRoot, name + ".jpg")
            gtPath = os.path.join(gtRoot, name + ".png")
            self.imgs.append(imgPath)
            self.gts.append(gtPath)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.imgs[index])
        img.convert("RGB")
        gt = Image.open(self.gts[index])
        gt.convert("L")
        return img, gt

    def __len__(self) -> int:
        return len(self.imgs)

class ClassDateset(data.Dataset):
    def __init__(self, root:str, flag = "train") -> None:
        self.imgs = []
        self.classes = []
        imgRoot = os.path.join(root, r"JPEGImages")
        annoRoot = os.path.join(root, r"Annotations")
        names = listNames(root, flag)
        for name in names:
            imgPath = os.path.join(imgRoot, name + ".jpg")
            annoPath = os.path.join(annoRoot, name + ".xml")
            self.imgs.append(imgPath)
            self.classes.append(self._parseXml(annoPath))
    
    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.imgs[index])
        img.convert("RGB")
        c = self.classes[index]
        return img, c

    def __len__(self) -> int:
        return len(self.imgs)

    def _parseXml(self, annoPath: str) -> str:
        root = ET.parse(annoPath).getroot()
        return root.find("object").find("name").text


if __name__ == "__main__":
    root = r"C:\Users\尹旭\Desktop\SHIPdevkit\VOC2012"
    segset = SegDataset(root, flag="test")
    print(len(segset))
    img, gt = segset[300]
    import numpy as np
    img = np.array(img)
    gt = np.array(gt)
    print(img)
    print(gt)
    
    clsset = ClassDateset(root)
    print(len(clsset))
    img2, c = clsset[3000]
    img2 = np.array(img2)
    print(img2)
    print(c)