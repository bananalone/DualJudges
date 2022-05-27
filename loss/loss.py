import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
    
    def __init__(self, reduction = "mean"):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction = reduction)


    def forward(self, predict, target):
        """

        """
        predict = F.log_softmax(predict, dim=1)
        target = F.softmax(target, dim=1)
        return self.kl(predict, target)


class DropMSELoss(nn.Module):
    
    def __init__(self, drop = 0.5, reduction = "mean"):
        super().__init__()
        self.drop = drop
        self.mse = nn.MSELoss(reduction = reduction)
        
    def forward(self, predict, target):
        rand = torch.rand(target.size(-2), target.size(-1))
        mask = torch.where(rand > self.drop, torch.tensor(1.0), torch.tensor(0.0))
        mask = mask.to(predict.device)
        predict = mask * predict
        target = mask * target
        return self.mse(predict, target)


class PFeedbackLoss(nn.Module):
    
    def __init__(self, alpha = 1.0):
        """训练PFeedback模块的损失函数

        Args:
            alpha (float, optional): in (0, 1). Defaults to 1.0.
            alpha 越接近1，input1和input2越不同
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor):
        pred1 = F.softmax(input1, dim=1)
        pred2 = F.softmax(input2, dim=1)
        loss = F.nll_loss(torch.log(pred1), target) + F.nll_loss(torch.log(pred2), target) + (- self.alpha) * F.mse_loss(pred1, pred2)
        return loss


class AdaptiveWeightedLoss(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.ceLoss = nn.CrossEntropyLoss(reduction = 'none')
    
    def forward(self, input: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor):
        input1 = F.softmax(input1.detach(), dim=1)
        input2 = F.softmax(input2.detach(), dim=1)
        # 伪标签
        pseudoLabel = torch.argmax(input1 + input2, dim=1)
        # 权重
        pred1 = torch.argmax(input1, dim=1)
        pred2 = torch.argmax(input2, dim=1)
        interMask = (pred1 > 0) & (pred1 == pred2)
        unionMask = (pred1 > 0) | (pred2 > 0)
        adaw = (torch.sum(interMask, dim=(1, 2), dtype=torch.float)) / (torch.sum(unionMask, dim=(1, 2), dtype=torch.float) + 1e-6)

        out = torch.mean(self.ceLoss(input, pseudoLabel), dim=(1, 2))
        loss = torch.mean(adaw * out)
        return loss


class PFeedbackLoss1(nn.Module):
    
    def __init__(self, weight: typing.Union[torch.Tensor, None] = None, reduction = 'mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, reduction = reduction)

    def getTarget(self, input: torch.Tensor, target: torch.Tensor):
        """
        从predict 和 target中生成新标签
        args:
            input: 网络预测值，范围 0-1
            target: pFeedback模块预测值，范围 0-1
        return:
            target: 新语义分割标签
        """
        input = F.softmax(input.detach(), dim=1)
        target = F.softmax(target.detach(), dim=1)
        target = torch.argmax(input + target, dim=1)
        return target

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        PFeedback loss
        args:
            predict: 网络预测值
            target: pFeedback模块预测值
        return:
            loss
        """
        target = self.getTarget(input, target)
        return self.loss(input, target)


if __name__ == "__main__":
    # a = torch.tensor(
    #     [[1, 2, 3],
    #     [1, 2, 3]]
    # )
    # ans = torch.sum(a, dim=0)
    # print(ans)
    # ans = torch.sum(a, dim=1)
    # print(ans)
    # ans = torch.sum(a, dim=[0, 1])
    # print(ans)
    # kl = KLDivLoss()
    # size = (4, 10, 8, 8)
    # a = torch.rand(*size)
    # b = torch.rand(*size)
    # softmax =  nn.Softmax(dim=1)
    # l = kl(softmax(a), softmax(b))
    # print(l)

    dropMSELoss = DropMSELoss()
    a = torch.randn(4, 11, 256, 256)
    b = torch.randn(4, 11, 256, 256)
    l = dropMSELoss(a, b)
    print(l)