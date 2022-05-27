import os
import time

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim

from config import Config, parseConfig
from log import initLogging, log
from data import ImageLabeledSet, PixelLabeledSet, BiPixelLabeledSet, DeviceDataloader
from net import FreezeModel, SegNet, DeepLabv3_plus, PFeedback
from loss import KLDivLoss, DropMSELoss, PFeedbackLoss, AdaptiveWeightedLoss

from train import trainPFeedBack, trainModel, supTrainModel
from test import IouMetric
from utils import getColorMap, initSave, batchSavePred, batchTest
from engine import Engine

# 初始化时间
t = time.strftime(r'%Y-%m-%d %H-%M-%S', time.localtime())

# 初始化控制引擎
engine = Engine()

# 初始化配置
cfg = Config()
parseConfig("./conf/config_deeplabv3-plus.ini", cfg)

# 初始化日志
initLogging(os.path.join(cfg.log.root, t + '.log'), cfg.log.log_fmt, cfg.log.date_fmt)

# 初始化训练数据
seed = 1
className = ['background', 'jyj', 'qwc', '3-tc_qzc_sag', '4_yyc', 'hwj', 'lqt', 'myyyc', 'qt', 'qzj', 'slj']
imgLabeledSet = ImageLabeledSet(cfg.data.root, className[1:], usage=cfg.data.usage, rest=True, seed=seed)
pxlLabeledSet = PixelLabeledSet(cfg.data.root, usage = cfg.data.usage, rest=False, seed=seed)

imgLabeledLoader = DeviceDataloader(data.DataLoader(imgLabeledSet, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True), cfg.train.device) 
pxlLabeledLoader =  DeviceDataloader(data.DataLoader(pxlLabeledSet, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True), cfg.train.device) 

# 初始化测试数据
pxlLabeledSetForTest = PixelLabeledSet(cfg.data.root, flag="test")
pxlLabeledLoaderForTest = DeviceDataloader(data.DataLoader(pxlLabeledSetForTest, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, drop_last=True), cfg.train.device)

# 初始化网络模型
segModel1 = DeepLabv3_plus(nInputChannels=3, n_classes=cfg.data.num_classes, os=16, pretrained=False, _print=True)
segModel1 = FreezeModel(segModel1)
segModel1.to(cfg.train.device)

segModel2 = DeepLabv3_plus(nInputChannels=3, n_classes=cfg.data.num_classes, os=16, pretrained=False, _print=True)
segModel2 = FreezeModel(segModel2)
segModel2.to(cfg.train.device)

segModel = DeepLabv3_plus(nInputChannels=3, n_classes=cfg.data.num_classes, os=16, pretrained=False, _print=True)
segModel = FreezeModel(segModel)
segModel.to(cfg.train.device)

pFeedback = PFeedback(segModel1, segModel2)
pFeedback = FreezeModel(pFeedback)
pFeedback.to(cfg.train.device)

# 初始化损失函数
ceLoss = nn.CrossEntropyLoss()
ceLoss.to(cfg.train.device)

klLoss = KLDivLoss()
klLoss.to(cfg.train.device)

dropMSELoss = DropMSELoss()
dropMSELoss.to(cfg.train.device)

pFeedbackLoss = PFeedbackLoss(alpha=1.0)
pFeedbackLoss.to(cfg.train.device)

adaptiveWeightedLoss = AdaptiveWeightedLoss()
adaptiveWeightedLoss.to(cfg.train.device)

# 初始化优化器
pFeedbackOptimizer = optim.Adam(filter(lambda p: p.requires_grad, pFeedback.parameters()), lr=cfg.train.lr)
segModelOptimizer = optim.Adam(filter(lambda p: p.requires_grad, segModel.parameters()), lr=cfg.train.lr)

# 初始化度量器, 模型保存路径, 保存器
model1Metric = IouMetric(className)
model2Metric = IouMetric(className)
modelMetric = IouMetric(className)

saveModelRoot = os.path.join(cfg.train.save_path, t)
if not os.path.exists(saveModelRoot):
    os.mkdir(saveModelRoot)

saveModel1To = os.path.join(saveModelRoot,  "{model}-1.pth".format(model=cfg.train.model_name, t=t))
saveModel2To = os.path.join(saveModelRoot,  "{model}-2.pth".format(model=cfg.train.model_name, t=t))
saveModelTo = os.path.join(saveModelRoot,  "{model}.pth".format(model=cfg.train.model_name, t=t))
# saveModel1To = r"out/models/2021-12-29 21-58-54/deeplabv3-plus-1.pth"
# saveModel2To = r"out/models/2021-12-29 21-58-54/deeplabv3-plus-2.pth"
# saveModelTo = r"out\models\pfeedback\segnet-pfeedback 2021-12-01 21-01-14.pth"

model1Saver = initSave(segModel1, saveModel1To)
model2Saver = initSave(segModel2, saveModel2To)
modelSaver = initSave(segModel, saveModelTo)

# 记录训练配置
@engine.step(priority=0)
@log("info", stdout=True)
def outConfig(step):
    if step == 1:
        return {
            "log" : cfg.log.__dict__,
            "data" : cfg.data.__dict__,
            "train" : cfg.train.__dict__,
            "test" : cfg.test.__dict__
        }


# 训练
stage1_start = 1
stage1_end = stage1_start - 1 + 100

@engine.step(priority=1)
@log("info", stdout=False)
def stage1(step):
    global pFeedbackOptimizer
    if step == stage1_start:
        pFeedback.freeze(freeze=False)
        pFeedbackOptimizer = optim.Adam(filter(lambda p: p.requires_grad, pFeedback.parameters()), lr=cfg.train.lr)
    if stage1_start <= step <= stage1_end:
        pFeedback.train(True)
        trainPFeedBack(pxlLabeledLoader, pFeedback, pFeedbackLoss, pFeedbackOptimizer)
        return {
            "stage1[{}]".format(step): "sup train"
        }


stage2_start = stage1_end + 1
stage2_end = stage2_start - 1 + 30

@engine.step(priority=2)
@log("info", stdout=False)
def stage2(step):
    global segModelOptimizer
    if step == stage2_start:
        segModel1.load_state_dict(torch.load(saveModel1To))
        segModel2.load_state_dict(torch.load(saveModel2To))
        pFeedback.freeze()
        segModelOptimizer = optim.Adam(filter(lambda p: p.requires_grad, segModel.parameters()), lr=cfg.train.lr)
    if stage2_start <= step <= stage2_end:
        pFeedback.train(False)
        segModel.train(True)
        trainModel(imgLabeledLoader, segModel, pFeedback, adaptiveWeightedLoss, segModelOptimizer)
        supTrainModel(pxlLabeledLoader, segModel, ceLoss, segModelOptimizer)
        return {
            "stage2[{}]".format(step): "semi-sup train".format(step)
        }



# 测试并保存最优模型
@engine.step(priority=5)
@log("info", stdout=True)
def test_model(step):
    segModel1.train(False)
    segModel2.train(False)
    segModel.train(False)
    if  stage1_start <= step <= stage1_end:
        batchTest(pxlLabeledLoaderForTest, segModel1, model1Metric)
        mIoU1 = model1Metric.calc()
        batchTest(pxlLabeledLoaderForTest, segModel2, model2Metric)
        mIoU2 = model2Metric.calc()
        rets = {
            "test for step": step,
            "segModel1": mIoU1,
            "segModel2": mIoU2,
        }
        if model1Saver(mIoU1[0]):
            rets["saveModel1To"] = saveModel1To
        if model2Saver(mIoU2[0]):
            rets["saveModel2To"] = saveModel2To
        return rets
    elif stage2_start <= step <= stage2_end:
        batchTest(pxlLabeledLoaderForTest, segModel, modelMetric)
        mIoU = modelMetric.calc()
        rets = {
            "test for step": step,
            "segModel": mIoU,
        }
        if modelSaver(mIoU[0]):
            rets["saveModelTo"] = saveModelTo
        return rets


# 保存可视化预测结果
colorMap = getColorMap(cfg.data.num_classes)

@engine.step(priority=6)
@log("info", stdout=True)
def saveResult(step):
    if step == stage2_end + 1:
        segModel1.load_state_dict(torch.load(saveModel1To))
        segModel1.to(cfg.train.device)
        segModel1.train(False)
        segModel2.load_state_dict(torch.load(saveModel2To))
        segModel2.to(cfg.train.device)
        segModel2.train(False)
        segModel.load_state_dict(torch.load(saveModelTo))
        segModel.to(cfg.train.device)
        segModel.train(False)
        saveTo = os.path.join(cfg.test.save_path, t)
        if not os.path.exists(saveTo):
            os.mkdir(saveTo)
        models = {
            "segModel1": segModel1,
            "segModel2": segModel2,
            "segModel": segModel,
        }
        batchSavePred(saveTo, pxlLabeledLoaderForTest, models, colorMap)
        return {
            "save result to" : saveTo
        }


if __name__ == "__main__":
    engine.run(1, 1 + stage2_end)
