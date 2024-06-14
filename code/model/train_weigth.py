import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

from model import Model, ComplexModel, Net1, Net2
from datasets import ComplexDataset_train, collate_fn_train
from utils import preprocess_train


def main():

    # 加载数据
    folder_path = "../data/trainset"
    data, labels = preprocess_train(folder_path)
    category_counts = [0] * 11

    # 划分数据集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.05, random_state=42)

    ros = RandomOverSampler(random_state=42)
    # train_data_resampled, train_labels_resampled = ros.fit_resample(train_data, train_labels)
    # print(len(train_data_resampled))

    train_dataset = ComplexDataset_train(train_data, train_labels)
    val_dataset = ComplexDataset_train(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_train)
    for _, batch_targets in train_loader:
        for target in batch_targets:
            category_counts[int(target)] += 1

    # 定义模型
    # model = ComplexModel()
    model = Net2()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型
    num_epochs = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    max_acc = 0.85

    losses = []  # 用于收集每个epoch的平均损失

    flag = False

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)

        flag = True

        # 验证集上的评估
        category_right = [0] * 11
        category_eval_counts = [0] * 11
        model.eval()
        with torch.no_grad():
            total_correct = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                for i in range(targets.size(0)):
                    if predicted[i] == targets[i]:
                        category_right[targets[i]] += 1
                    category_eval_counts[targets[i]] += 1

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_train)
        with torch.no_grad():
            total_correct_train = 0
            errors = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                errors.extend((predicted != targets).float())
                total_correct_train += (predicted == targets).sum().item()
            sample_weights = [error / (train_dataset.__len__() - total_correct_train) for error in errors]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn_train)

        accuracy = total_correct / len(val_dataset)
        accuracy_train = total_correct_train / len(train_dataset)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}, Train Accuracy: {accuracy_train}')
        print(f'{category_right[0]} {category_right[1]} {category_right[2]} {category_right[3]} {category_right[4]} {category_right[5]} {category_right[6]} {category_right[7]} {category_right[8]} {category_right[9]} {category_right[10]}')
        print(f'{category_eval_counts[0]} {category_eval_counts[1]} {category_eval_counts[2]} {category_eval_counts[3]} {category_eval_counts[4]} {category_eval_counts[5]} {category_eval_counts[6]} {category_eval_counts[7]} {category_eval_counts[8]} {category_eval_counts[9]} {category_eval_counts[10]}')
        print(f'{category_counts[0]} {category_counts[1]} {category_counts[2]} {category_counts[3]} {category_counts[4]} {category_counts[5]} {category_counts[6]} {category_counts[7]} {category_counts[8]} {category_counts[9]} {category_counts[10]}')
        if accuracy >= max_acc:
            torch.save(model, f'../checkpoint/model_cnn_{int(accuracy * 10000)}.pth')
            max_acc = accuracy

    # 绘制损失随epoch变化的图
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, num_epochs+1), losses, marker='o', linestyle='-', color='b')
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.savefig('training_loss.png')

if __name__ == "__main__":
    main()