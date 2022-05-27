import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from tqdm import tqdm


def trainPFeedBack(dataloader: data.DataLoader, pFeedback: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer):
    """
    使用像素级标签训练正反馈模块
    """
    batchs = tqdm(dataloader, ncols=100)
    for batch in batchs:
        input, target = batch
        out1, out2 = pFeedback(input)
        loss = criterion(out1, out2, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchs.set_description(desc="loss: {:.6f}".format(loss.item()))


def trainModel(dataloader: data.DataLoader, segModel: nn.Module, pFeedback: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer):
    """
    使用pFeedback模块训练分割模块
    """
    batchs = tqdm(dataloader, ncols=100)
    for batch in batchs:
        input, target = batch
        out1, out2 = pFeedback(input)
        
        pred = segModel(input)
        loss = criterion(pred, out1, out2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchs.set_description(desc="loss: {:.6f}".format(loss.item()))


def supTrainModel(dataloader: data.DataLoader, segModel: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer):
    """强监督训练分割模块

    Args:
        dataloader (data.DataLoader): [description]
        segModel (nn.Module): [description]
        criterion (nn.Module): [description]
        optimizer (optim.Optimizer): [description]
    """
    batchs = tqdm(dataloader, ncols=100)
    for batch in batchs:
        input, target = batch
        out = segModel(input)
    
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchs.set_description(desc="loss: {:.6f}".format(loss.item()))