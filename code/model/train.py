import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import Model, ComplexModel
from datasets import ComplexDataset_train, collate_fn_train
from utils import preprocess_train


def main():

    # 加载数据
    folder_path = "../data/trainset"
    data, labels = preprocess_train(folder_path)

    # 划分数据集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, random_state=42)

    train_dataset = ComplexDataset_train(train_data, train_labels)
    val_dataset = ComplexDataset_train(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_train)

    # 定义模型
    model = ComplexModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型
    num_epochs = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    max_acc = 0.7

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证集上的评估
        model.eval()
        with torch.no_grad():
            total_correct = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()

        accuracy = total_correct / len(val_dataset)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}')
        if accuracy >= max_acc:
            torch.save(model, f'../checkpoint/model_{int(accuracy * 10000)}.pth')
            max_acc = accuracy

if __name__ == "__main__":
    main()