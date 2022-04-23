import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values

    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none'), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        # m_list 即为论文中标注的gamma_j
        # 这里的常数直接给定为0.5/np.max(m_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        print(m_list)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        # 把target也由shape为(n_sample,1)的格式变为(n_sample. 10)的格式
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        # self.m_list[None, :]shape=(1,10)
        # index_float.transpose(0,1) shape=(10, n_samples)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # 得到的batch_m shape为(n_samples, 1)，表示的是原文公式中的delta_y项
        batch_m = batch_m.view((-1, 1))
        # 进行广播
        x_m = x - batch_m
        # 仅仅保留x每一行中下标为target对应索引的项——z_y-delta_y
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

class LDAMLoss_CB(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, s=30, keep_prob=0.4, epoch=1, beta=0.999, reg=0.001):
        super(LDAMLoss_CB, self).__init__()
        # m_list 即为论文中标注的gamma_j
        # 这里的常数直接给定为0.5/np.max(m_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))

        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.keep_prob = keep_prob
        print("m_list")
        print(m_list)

        assert s > 0
        self.s = s
        self.epoch = epoch

        # cb setup
        effective_num = 1.0 - np.power(beta, cls_num_list)
        self.weights = (1.0 - beta) / np.array(effective_num)
        self.weights = torch.cuda.FloatTensor(self.weights / np.sum(self.weights) * len(cls_num_list))
        print(self.weights)
        self.no_of_classes = len(cls_num_list)
        self.reg = reg

    def forward(self, x, target):
        # x shape=(num_samples, num_classes)
        # target shape=(num_samples, 1)
        index = torch.zeros_like(x, dtype=torch.uint8)
        # 把target也由shape为(n_sample,1)的格式变为(n_sample. 10)的格式
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        # self.m_list[None, :]shape=(1,10)
        # index_float.transpose(0,1) shape=(10, n_samples)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # 得到的batch_m shape为(n_samples, 1)，表示的是原文公式中的delta_y项
        batch_m = batch_m.view((-1, 1))
        # 进行广播
        x_m = x - batch_m
        # 仅仅保留x每一行中下标为target对应索引的项——z_y-delta_y
        output = torch.where(index, x_m, x)

        mask = pred(x).eq_(target)

        o, _ = torch.max(x, 1)

        mask_ = mask * torch.norm(1-(o.unsqueeze(1).sub(x)), 2, dim=1)
        if(self.epoch>=100):
            loss = F.cross_entropy(self.s * output, target, weight=self.weights, reduce=False)
            loss = torch.mean(mask_ * loss)*self.reg

        else:
            loss = F.cross_entropy(self.s * output, target, weight=self.weights)

        return loss



if __name__ == '__main__':
    x = torch.tensor(np.random.randn(64, 70, 9))
    label = torch.tensor(np.random.randint(0, 9, size=(64, 70)), dtype=torch.long).squeeze(1)

    print(label)
    loss =CSLoss()
    print(loss(x, label.long()))