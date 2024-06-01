import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(30720, 256)
        self.fc2 = nn.Linear(256, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        # 比原模型更复杂的结构，添加更多的全连接层和Dropout以及BatchNorm层
        self.fc1 = nn.Linear(30720, 1024)  # 从30720维降至1024维
        self.bn1 = nn.BatchNorm1d(1024)  # 批量归一化
        self.dropout1 = nn.Dropout(0.5)  # Dropout层，防止过拟合
        
        self.fc2 = nn.Linear(1024, 512)  # 进一步降维
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 256)  # 降至256维
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(256, 11)  # 输出层，输出11个类别的得分

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # ReLU激活函数 + 批量归一化
        x = self.dropout1(x)  # Dropout
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)  # 最后一层不做激活，直接输出
        return x