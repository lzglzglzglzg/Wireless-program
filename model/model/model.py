from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(30720, 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class mlp(nn.Module):
    def __init__(self, num_classes=11):
        super(mlp, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(30720, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x


class EnhancedModel(nn.Module):
    def __init__(self):
        super(EnhancedModel, self).__init__()
        self.fc0 = nn.Linear(6800, 1024)
        # 添加Batch Normalization
        self.bn0 = nn.BatchNorm1d(1024)
        # 输入层到隐藏层1
        self.fc1 = nn.Linear(1024, 512)
        # 添加Batch Normalization
        self.bn1 = nn.BatchNorm1d(512)
        # 隐藏层1到隐藏层2
        self.fc2 = nn.Linear(512, 256)
        # 添加Batch Normalization
        self.bn2 = nn.BatchNorm1d(256)
        # 隐藏层2到隐藏层3
        self.fc3 = nn.Linear(256, 128)
        # 添加Batch Normalization
        self.bn3 = nn.BatchNorm1d(128)
        # 隐藏层3到输出层
        self.fc4 = nn.Linear(128, 11)
        # 添加Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # 最后一层不用激活函数，直接输出
        return x


class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(32, 64, 11, 1, 5)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            # nn.Linear(491520,1024),
            # nn.Linear(32768, 1024),
            # pca 6800
            # nn.Linear(119040, 1024),
            # pca 9600
            nn.Linear(153600, 1024),
            # pca 11000
            # nn.Linear(175872, 1024),
            # pca 4096
            # nn.Linear(65536, 1024),
            # pca 2048
            # nn.Linear(32768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 11),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        # print(f"Input size: {x.shape}")
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.bn4(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # print(f"After view: {x.shape}")
        x = self.linear_unit(x)
        return x


class CNN2(nn.Module):

    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(32, 64, 11, 1, 5)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            # pca 6800
            # nn.Linear(108800, 1024),
            # pca 9600
            nn.Linear(153600, 1024),
            # pca 4096
            # nn.Linear(65536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 11),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # print(f"After view: {x.shape}")
        x = self.linear_unit(x)
        return x


class CNN3(nn.Module):

    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(32, 64, 11, 1, 5)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            # pca 6800
            # nn.Linear(108800, 1024),
            # pca 9600
            nn.Linear(153600, 1024),
            # pca 4096
            # nn.Linear(65536, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 11),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x))
        x = self.bn4(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear_unit(x)
        return x
