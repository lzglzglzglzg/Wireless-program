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

from model import Model, ComplexModel
from datasets import ComplexDataset_train, collate_fn_train
from utils import preprocess_train_main

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # 加载数据
    folder_path = "../../data/trainset"
    data, labels = preprocess_train_main(folder_path)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用决策树作为弱分类器
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Adaboost模型
    model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=1.0, random_state=42, algorithm='SAMME')
    print("train start!")
    model.fit(X_train, y_train)
    print("train end!")

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-0', '0'], yticklabels=['Non-0', '0'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_Matrix.png')

    # 错误率曲线
    error_rate = []
    for i, y_pred_iter in enumerate(model.staged_predict(X_train)):
        error_rate.append(1 - accuracy_score(y_train, y_pred_iter))

    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(error_rate) + 1), error_rate, marker='o')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.title('Training Error Rate vs Number of Estimators')
    plt.savefig('Error_Rate.png')


if __name__ == "__main__":
    main()