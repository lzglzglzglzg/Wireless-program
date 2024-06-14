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
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1(self.fc1(x)))  # ReLU激活函数 + 批量归一化
        x = self.dropout1(x)  # Dropout
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)  # 最后一层不做激活，直接输出
        return x
    
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, dilation=2)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.dropout = nn.Dropout(0.5)
        
        # The following line will be updated once we determine the input size dynamically
        # self.fc1 = None
        self.fc1 = nn.Linear(61408, 1024)  # 从30720维降至1024维
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
        o_x = x
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)

        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x)) 
        x = self.max_pool2(x)

        x = self.dropout(x)
        o_x = torch.flatten(o_x, 1)
        x = torch.flatten(x, 1)  # Flatten starting at dimension 1
        # x = torch.cat((o_x, x), dim=1)
        # print(x.shape)
        
        x = F.relu(self.bn1(self.fc1(x)))  # ReLU激活函数 + 批量归一化
        x = self.dropout1(x)  # Dropout
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)  # 最后一层不做激活，直接输出
        return x
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=7, dilation=2)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, dilation=2)
        self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.dropout = nn.Dropout(0.5)
        
        # The following line will be updated once we determine the input size dynamically
        # self.fc1 = None
        self.fc1 = nn.Linear(61344, 1024)  # 从30720维降至1024维
        self.bn1 = nn.BatchNorm1d(1024)  # 批量归一化
        self.dropout1 = nn.Dropout(0.5)  # Dropout层，防止过拟合

        self.fc2 = nn.Linear(1024, 512)  # 进一步降维
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 256)  # 降至256维
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(1024, 11)  # 输出层，输出11个类别的得分
        
    def forward(self, x):
        o_x = x
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)

        x = self.dropout(x)
        o_x = torch.flatten(o_x, 1)
        x = torch.flatten(x, 1)  # Flatten starting at dimension 1
        # x = torch.cat((o_x, x), dim=1)
        # print(x.shape)
        
        x = F.relu(self.bn1(self.fc1(x)))  # ReLU激活函数 + 批量归一化
        x = self.dropout1(x)  # Dropout
        
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout2(x)
        
        # x = F.relu(self.bn3(self.fc3(x)))
        # x = self.dropout3(x)
        
        x = self.fc4(x)  # 最后一层不做激活，直接输出
        return x