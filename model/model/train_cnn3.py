import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import Model, EnhancedModel, CNN1, CNN2, CNN3
# from imblearn.over_sampling import ADASYN, SMOTE
from collections import Counter
# from focalloss import FocalLoss
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.decomposition import PCA
import csv

# from tsaug import AddNoise, Drift, TimeWarp


def preprocess(folder_path):
    labels = []
    data = []
    all_complex_data = []
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        if filename.endswith(".bin"):
            match = re.search(r'label_(\d+)_', filename)
            if match:
                label = int(match.group(1))
            else:
                continue
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                labels.append(label)
                # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.frombuffer(
                    data_row_bin, dtype=np.float16)
                real_part = data_row_float16[::2]
                imaginary_part = data_row_float16[1::2]
                complex_data = real_part + 1j * imaginary_part
                complex_data = np.array(complex_data)
                # magnitude_spectrum = np.abs(complex_data)
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
                all_complex_data.append(complex_data)
    return all_complex_data, data, labels


def preprocess_test(folder_path):
    # labels = []
    data = []
    cnt = 0
    files = os.listdir(folder_path)
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    for file in sorted_files:
        file_path = os.path.join(folder_path, file)
        cnt += 1
        if file_path.endswith(".bin"):
            with open(file_path, 'rb') as file:
                data_row_bin = file.read()
                # labels.append(label)
                # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.frombuffer(
                    data_row_bin, dtype=np.float16)
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data


# 加载数据
folder_path = "./data/trainset"
all_complex_data, data, labels = preprocess(folder_path)

# 读取测试数据，修改测试集路径
folder_path_test = "./data/z_test/"
test_data = preprocess_test(folder_path_test)
# print(len(test_data))
# 测试数据接到训练数据后

data.extend(test_data)

if not os.path.isfile("pca_9600.npz"):
    # 降维，维度改为 len(train_data) + len(test_data)
    pca = PCA(n_components=9600)
    data = pca.fit_transform(data)
    print(data[0].shape)

    # 保存，修改train_data=data[:5400]，test_data=data[5400:]
    np.savez('pca_9600.npz',
            train_data=data[:5400], test_data=data[5400:])

# 读取.npz文件
loaded_arrays = np.load('pca_9600.npz')

# 访问加载的数组
# print(loaded_arrays['train_data'])
# print(loaded_arrays['test_data'])
data = loaded_arrays['train_data']
print(f'length of dataset: {len(data)}')

# data = data[:5400]

# 划分数据集
train_data, val_data, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.1, random_state=42)

print(f'data length: {train_data[0].shape}')

print(f"train dataset shape: {Counter(train_labels)}")

print(f"val dataset shape: {Counter(val_labels)}")


class ComplexDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cnn加 .unsqueeze(0)
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32),
                  'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample


train_dataset = ComplexDataset(train_data, train_labels)
val_dataset = ComplexDataset(val_data, val_labels)


def collate_fn(batch):
    features = []
    labels = []
    for _, item in enumerate(batch):
        features.append(item['data'])
        labels.append(item['label'])
    return torch.stack(features, 0), torch.stack(labels, 0)


# 定义模型

model = CNN3()

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle=False, collate_fn=collate_fn)


# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(class_num=11)
# lr = 0.001
optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-5)
clr = CosineAnnealingLR(optimizer, T_max=32)  # 使用余弦退火算法改变学习率
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 设定优优化器更新的时刻表

# 训练模型
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(model)
model.to(device)

# czz
# model = nn.DataParallel(model)  # 包装你的模型以进行并行计算

best_accuracy = 0.9
losses = []
for epoch in range(num_epochs):
    # scheduler.step()
    model.train()
    epoch_losses = []
    total_correct_train = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_correct_train += (predicted == targets).sum().item()
        # outputs = torch.softmax(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        clr.step()  # 学习率迭代
        epoch_losses.append(loss.item())

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(epoch_loss)

    # 验证集上的评估
    model.eval()
    with torch.no_grad():
        n_classes = 11
        incorrect_classifications = [0] * n_classes
        total_correct = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            # 计算错误分类
            incorrect_predictions = (predicted != targets)
            for i in range(len(targets)):
                if incorrect_predictions[i]:
                    # 对错误分类的类别进行计数
                    incorrect_classifications[targets[i].item()] += 1
    # 打印每个类别的错误分类数量
    for i, count in enumerate(incorrect_classifications):
        print(f"Class {i}: Incorrectly classified {count} times")

    accuracy_train = total_correct_train / len(train_dataset)
    accuracy = total_correct / len(val_dataset)
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}, Val Loss: {val_loss}, Train Accuracy: {accuracy_train}, Train Loss: {loss}')

    if not os.path.isdir("models_saved_cnn3_pca_9600"):
        os.makedirs("models_saved_cnn3_pca_9600")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(
            model, f'./models_saved_cnn3_pca_9600/model_{int(accuracy * 10000)}.pth')

# 绘制损失随epoch变化的图
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs+1), losses, marker='o', linestyle='-', color='b')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig('training_loss.png')
