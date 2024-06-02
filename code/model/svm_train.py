import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from model import Model, ComplexModel
from datasets import ComplexDataset_train, collate_fn_train
from utils import preprocess_train


def main():

    # 加载数据
    folder_path = "../data/trainset"
    data, labels = preprocess_train(folder_path)

    # 划分数据集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, random_state=42)

    # 数据标准化对于SVM很重要，因为它使用了距离的概念
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data)
    X_test = scaler.transform(val_data)

    # 使用默认的参数创建SVM分类器
    # 这将会使用一对一（OvO）方法进行多分类
    clf = svm.SVC(decision_function_shape='ovo')

    # 训练模型
    clf.fit(X_train, train_labels)

    # 预测测试集
    y_pred = clf.predict(X_test)

    # 计算精度并显示分类报告
    print(accuracy_score(val_labels, y_pred))
    print(classification_report(val_labels, y_pred))

if __name__ == "__main__":
    main()