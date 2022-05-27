from typing import Any, List
import random

import numpy as np
from torchvision import transforms
import torch

from data.base import SegDataset
from data.base import ClassDateset


class ImageLabeledSet(ClassDateset):
    def __init__(self, root: str, className: List[str], size=(256, 256), flag="train", usage=1, rest=False, seed=1) -> None:
        """VOC格式分类数据集

        Args:
            root (str): 数据路径
            className (List[str]): className是类别列表，不包含background
            size (tuple, optional): 图像大小. Defaults to (256, 256).
            flag (str, optional): 为"train"时获取训练集，其他时为测试集. Defaults to "train".
            usage (int, optional): 所使用的挑选出的数据量，范围是0-1. Defaults to 1.
            rest (bool, optional): 是否使用剩余未挑选的数据，是则会返回 (1-usage) * len(data) 的数据. Defaults to False.
            seed (int, optional): 伪随机种子. Defaults to 1.
        """
        super().__init__(root, flag=flag)
        self.className = className
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.cls_transform = torch.tensor
        length = len(self.imgs)
        self.ids = list(range(length))
        random.seed(seed)
        random.shuffle(self.ids)
        if not rest:
            self.ids = self.ids[:int(length * usage)]
        else:
            self.ids = self.ids[int(length * usage):]

    def __getitem__(self, index: int) -> Any:
        img, c = super().__getitem__(self.ids[index])
        img = self.img_transform(img)
        c =  self.cls_transform(self.className.index(c))
        return img, c

    def __len__(self) -> int:
        return len(self.ids)


class PixelLabeledSet(SegDataset):
    def __init__(self, root: str, size=(256, 256), flag="train", usage= 1, rest=False, seed=1) -> None:
        """VOC格式语义分割数据集

        Args:
            root (str): 数据路径
            size (tuple, optional): 缩放图像大小. Defaults to (256, 256).
            flag (str, optional): 为"train"时获取训练集，其他时为测试集. Defaults to "train".
            usage (int, optional): 所使用的挑选出的数据量，范围是0-1. Defaults to 1.
            rest (bool, optional): 是否使用剩余未挑选的数据，是则会返回 (1-usage) * len(data) 的数据. Defaults to False.
            seed (int, optional): 伪随机种子. Defaults to 1.
        """
        super().__init__(root, flag=flag)
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize(size)
        ])
        length = len(self.imgs)
        self.ids = list(range(length))
        random.seed(seed)
        random.shuffle(self.ids)
        if not rest:
            self.ids = self.ids[:int(length * usage)]
        else:
            self.ids = self.ids[int(length * usage):]

    def __getitem__(self, index: int) -> Any:
        img, gt = super().__getitem__(self.ids[index])
        img = self.img_transform(img)
        gt = self.gt_transform(gt)
        gt = torch.tensor(np.array(gt))
        return img, gt.long()

    def __len__(self) -> int:
        return len(self.ids)


class BiPixelLabeledSet(PixelLabeledSet):
    def __init__(self, root: str, num_classes: int, size=(256, 256), flag="train", usage=1) -> None:
        """[标签二值化的分割数据集，2d版的one hot，只区分前景和背景]

        Args:
            root (str): [description]
            num_classes (int): [分割数据集的总类别数，包含背景]
            size (tuple, optional): [description]. Defaults to (256, 256).
            flag (str, optional): [description]. Defaults to "train".
            usage (int, optional): [description]. Defaults to 1.
        """
        super().__init__(root, size=size, flag=flag, usage=usage)
        self.num_classes = num_classes
    
    def __getitem__(self, index: int) -> Any:
        img, gt = super().__getitem__(index)
        ones = torch.ones_like(gt)
        zeros = torch.zeros_like(gt)
        fore = torch.where(gt > 0, ones, zeros)
        bg = torch.where(gt == 0, ones, zeros).unsqueeze(dim=0)
        fore = fore.repeat([self.num_classes - 1, 1, 1])
        biGt = torch.cat((bg, fore), dim=0)
        return img, biGt.float()

    def __len__(self) -> int:
        return super().__len__()


if __name__ == "__main__":

    root = r"C:\Users\尹旭\Desktop\SHIPdevkit\VOC2012"
    className = ['background', 'jyj', 'qwc', '3-tc_qzc_sag', '4_yyc', 'hwj', 'lqt', 'myyyc', 'qt', 'qzj', 'slj']
    imgLabeledSet = ImageLabeledSet(root, className[1:])
    pxlLabeledSet = PixelLabeledSet(root, usage=1/32)
    biPixelLabeledSet = BiPixelLabeledSet(root, 11, usage=1/32)

    iSetMap = {
        "length": len(imgLabeledSet),
        "sample": imgLabeledSet[2000]
    }

    pSetMap = {
        "length": len(pxlLabeledSet),
        "sample": pxlLabeledSet[100],
        "label": pxlLabeledSet[100][1].cpu().numpy()
    }

    biSetMap = {
        "length": len(biPixelLabeledSet),
        "sample": biPixelLabeledSet[100],
        "label": biPixelLabeledSet[100][1].cpu().numpy()
    }

    print(iSetMap)
    print(pSetMap)
    print(biSetMap)

    from torch.utils import data
    from deviceloader import DeviceDataloader
    
    dataloader = data.DataLoader(imgLabeledSet, batch_size=4, shuffle=False, num_workers=4)
    deviceDataloader = DeviceDataloader(dataloader, "cuda:0")
    for batch in deviceDataloader:
        print(batch)
        break