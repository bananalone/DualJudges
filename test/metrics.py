from typing import List
import numpy as np


class Metric:
    def add(self, predict: np.ndarray, label: np.ndarray):
        raise NotImplementedError("add function is not implemented")
    
    def reset(self):
        raise NotImplementedError("reset function is not implemented")
    
    def calc(self):
        raise NotImplementedError("calc function is not implemented")
    

class IouMetric(Metric):
    def __init__(self, className: List[str]) -> None:
        self.className = className
        self.num_classes = len(className)
        self.hist = np.zeros(shape=(self.num_classes, self.num_classes))
    
    def __fast_hist(self, predict: np.ndarray, label: np.ndarray):
        '''
        输入： predict  预测分割图像展开的一维行向量
              label 标签图像展开的一维行向量
        输出： 统计TP、FN和TN的n*n的混淆矩阵
        '''
        mask = (predict >= 0) & (predict < self.num_classes)
        return np.bincount(self.num_classes * predict[mask].astype(int) + label[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
    
    def add(self, predict: np.ndarray, label: np.ndarray):
        self.hist += self.__fast_hist(predict.reshape(-1), label.reshape(-1))
        
    def reset(self):
        self.hist = np.zeros(shape=(self.num_classes, self.num_classes))
    
    def calc(self):
        iouForClass = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist))
        iouMap = dict()
        for i in range(self.num_classes):
            iouMap[self.className[i]] = iouForClass[i]
        mIoU = np.nanmean(iouForClass)
        return mIoU, iouMap