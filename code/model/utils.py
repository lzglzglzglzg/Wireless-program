import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def preprocess_train(folder_path):
    labels = []
    data = []
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
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data, labels

def preprocess_test(folder_path):
    data = []
    cnt = 0
    for filename in os.listdir(folder_path):
        cnt += 1
        if filename.endswith(".bin"):
            match = re.search(r'label_(\d+)_', filename)
            with open(os.path.join(folder_path, filename), 'rb') as file:
                data_row_bin = file.read()
                data_row_float16 = np.frombuffer(data_row_bin, dtype=np.float16)  # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data