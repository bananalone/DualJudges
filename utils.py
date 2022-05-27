import typing
from PIL import Image
import os
import math

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from test.metrics import Metric


def getColorMap(num_classes):
    assert 256 ** 3 >= num_classes >= 1
    color_map = []
    radix = math.ceil(num_classes ** (1 / 3))  # 进制
    rate = 255 // (radix - 1)
    for cls in range(num_classes):
        r = (cls % radix) * rate
        g = ((cls // radix) % radix) * rate
        b = (((cls // radix) // radix) % radix) * rate
        color_map.append([r, g, b])
    return np.array(color_map, dtype=np.uint8)


def initSave(segModel: nn.Module, saveTo: str):
    best = 0.0
    def save(score: float):
        nonlocal best
        if score > best:
            torch.save(segModel.state_dict(), saveTo)
            best = score
            return True
        return False
    return save


def predict(images: torch.Tensor, segModel: nn.Module, pFeedback: nn.Module):
    """
    image is size of 1 * 3 * H * W
    """
    decodings, encodings = segModel(images)
    finePreds, coarsePreds = pFeedback(encodings, decodings)
    finePreds = torch.argmax(finePreds, dim=1)
    coarsePreds = torch.argmax(coarsePreds, dim=1)
    return finePreds.cpu().numpy(), coarsePreds.cpu().numpy()


def savePred(pred: np.ndarray, saveTo: str, colorMap: np.ndarray):
    """
    保存预测图
    """
    predMap = Image.fromarray(pred.astype(np.uint8))
    predMap.convert('P')
    predMap.putpalette(colorMap)
    predMap.save(saveTo)


def batchSavePred(saveTo: str, dataloader: data.DataLoader, models: typing.Union[nn.Module, typing.Dict[str, nn.Module]], colorMap: np.ndarray):
    """
    批量生成预测结果并保存，原图保存在images目录下，分割标签保存在labels目录下
    
    Args:
        saveTo (str): 保存路径
        dataloader (data.DataLoader): 待生成分割图的数据集
        models (typing.Union[nn.Module, typing.Dict[str, nn.Module]]): 分割模型，若存在多个分割模型，需要用dict保存，key为模型名，value为分割模型，分割结果保存在以key命名的目录下
        colorMap (np.ndarray): 颜色映射值
    """
    if isinstance(models, nn.Module):
        models = {
            'model': models,
        }
    imagesRoot = os.path.join(saveTo, "images")
    labelsRoot = os.path.join(saveTo, "labels")

    if not os.path.exists(imagesRoot):
        os.mkdir(imagesRoot)
    if not os.path.exists(labelsRoot):
        os.mkdir(labelsRoot)

    predRoots = dict()
    for k in models:
        predRoot = os.path.join(saveTo, k)
        predRoots[k] = predRoot
        if not os.path.exists(predRoot):
            os.mkdir(predRoot)

    batchs = tqdm(dataloader, ncols=100)
    batchs.set_description("saving")
    bid = 0
    for batch in batchs:
        images, labels = batch
        for k in models:
            output = models[k](images)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            id = bid
            for pred in preds:
                savePred(pred, os.path.join(predRoots[k], str(id) + ".png"), colorMap)
                id += 1

        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        id = bid
        for image, label in zip(images, labels):
            image = (image - image.min()) / (image.max() - image.min())
            image = np.transpose(image, axes=[1,2,0])
            plt.imsave(os.path.join(imagesRoot, str(id) + ".jpg") , image)
            savePred(label, os.path.join(labelsRoot, str(id) + ".png"), colorMap)
            id += 1
    
        bid += len(images)


def batchTest(dataloader: data.DataLoader, segModel: nn.Module,metric: Metric):
    """
    批量测试并计算mIoU
    
    Args:
        dataloader (data.DataLoader): 测试集
        segModel (nn.Module): 分割模型
        metric (Metric): 度量器
    """
    metric.reset()
    batchs = tqdm(dataloader, ncols=100)
    batchs.set_description("testing")
    for batch in batchs:
        images, labels = batch
        labels = labels.cpu().numpy()
        output = segModel(images)
        preds = torch.argmax(output, dim=1).cpu().numpy()
        for pred, label in zip(preds, labels):
            metric.add(pred, label)
 