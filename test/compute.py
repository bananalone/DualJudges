import os
from PIL import Image

import numpy as np


def fast_hist(pred, label, n):
    '''
    输入： pred  预测分割图像展开的一维行向量
    label 标签图像展开的一维行向量
    n 包括背景在内的类别总数
    输出： 统计TP、FN和TN的n*n的矩阵
    '''
    k = (pred >= 0) & (pred < n)
    return np.bincount(n * pred[k].astype(int) + label[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    '''
    输入： fast_hist计算得到的n*n的统计矩阵
            label 标签图像展开的一维行向量
            n 包括背景在内的类别总数
    输出： 每一类的IOU
    '''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


# hist.sum(0)=按列相加  hist.sum(1)按行相加
'''
  compute_mIoU函数原始以CityScapes图像分割验证集为例来计算mIoU值的（可以根据自己数据集的不同更改类别数num_classes及类别名称name_classes），本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。主要留意mIoU指标的计算核心代码即可。
'''
def compute_mIoU(label_pred_path_list: list, classes: list):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    num_classes = len(classes) + 1  # 读取类别数目，这里是11类
    print('Num classes', num_classes)  # 打印一下类别数目
    name_classes = np.array(['background'] + classes, dtype=np.str)  # 读取类别名称
    hist = np.zeros((num_classes, num_classes))  # hist初始化为全零，在这里的hist的形状是[11, 11]

    for ind in range(len(label_pred_path_list)):  # 读取每一个（图片-标签）对
        label = np.array(Image.open(label_pred_path_list[ind][0]))  # 读取一张图像分割结果，转化成numpy数组
        pred = np.array(Image.open(label_pred_path_list[ind][1]))  # 读取一张对应的标签，转化成numpy数组
        if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), label_pred_path_list[ind][0],
                                                                                  label_pred_path_list[ind][1]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind,len(label_pred_path_list), 100 * np.mean(per_class_iu(hist))))
            print(per_class_iu(hist))

    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs


def extract_ids_from_text(ids_path: str) -> list:
    with open(ids_path, 'r') as f:
        ids = f.read().strip().split("\n")
    return ids


def create_label_pred_path_list(label_path, pred_path, ids_path):
    label_pred_path_list = []
    ids = extract_ids_from_text(ids_path)
    for id in ids:
        label = os.path.join(label_path, id+'.png')
        pred = os.path.join(pred_path, id+'.png')
        label_pred_path_list.append([label, pred])
    return label_pred_path_list