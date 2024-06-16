import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import re
import numpy as np
import torch.nn as nn
from model import Model, EnhancedModel, CNN1, CNN2, CNN3


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
            # match = re.search(r'label_(\d+)_', filename)
            # if match:
            #     label = int(match.group(1))
            # else:
            #     continue
            with open(file_path, 'rb') as file:
                data_row_bin = file.read()
                # labels.append(label)
                # 原始数据是float16，直接把二进制bin读成float16的数组
                data_row_float16 = np.frombuffer(
                    data_row_bin, dtype=np.float16)
                data_row_float16 = np.array(data_row_float16)
                data.append(data_row_float16)
    return data


# 修改为线下测试数据路径
folder_path = "./data/z_test/"
data = preprocess_test(folder_path)
# czz 加载numpy
loaded_arrays = np.load('pca_9600.npz')

# 访问加载的数组
# print(loaded_arrays['train_data'])
# print(loaded_arrays['test_data'])
data = loaded_arrays['test_data']
print(f'length of dataset: {len(data)}')


class ComplexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float32)}
        return sample


# 创建测试数据集实例
test_dataset = ComplexDataset(data)


def collate_fn(batch):
    features = []
    for _, item in enumerate(batch):
        features.append(item['data'])
    return torch.stack(features, 0)


# 构建test_loader
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, collate_fn=collate_fn)

# 加载模型, 修改模型数量
model_path = ['' for i in range(6)]
model = [CNN1(), CNN1(), CNN1(), CNN3(), CNN3(), CNN3()]
model_path[0] = './models_saved_cnn1_pca_9600/model_9851.pth'
model_path[1] = './models_saved_cnn1_pca_9600/model_9888.pth'
model_path[2] = './models_saved_cnn1_pca_9600/model_9925.pth'

model_path[3] = './models_saved_cnn3_pca_9600/model_9722.pth'
model_path[4] = './models_saved_cnn3_pca_9600/model_9759.pth'
model_path[5] = './models_saved_cnn3_pca_9600/model_9796.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(len(model)):
    model[i] = torch.load(model_path[i], map_location='cpu')
    model[i].eval()
    model[i].to(device)

print(model)
# 假设test_loader用于加载无标签的测试数据
predictions = []


with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        temp_output = []
        temp_pred = []
        for m in model:
            outputs = m(inputs)
            temp_output.append(outputs)
            _, pred = outputs.max(1)
            temp_pred.append(np.array(pred.cpu()))
        temp_pred = np.transpose(np.array(temp_pred))
        batch_final_labels = []
        for ii in range(len(temp_pred)):
            line_pred = temp_pred[ii, :]
            max_num = list(np.bincount(line_pred)).index(
                np.max(np.bincount(line_pred)))  # 求预测结果的众数，生成标签
            batch_final_labels.append(max_num)

        predictions.extend(batch_final_labels)


print(predictions)
df_predictions = pd.DataFrame({'Prediction': predictions})

# 将预测结果保存到CSV文件，提交时注意去除表头
i = 1
csv_output_path = '.' + f'/result_{i}.csv'
if os.path.exists(csv_output_path):
    i += 1
    csv_output_path = '.' + f'/result_{i}.csv'
df_predictions.to_csv(csv_output_path, index=False,
                      header=False)  # index=False避免将索引写入CSV文件

print(f'Predictions have been saved to {csv_output_path}')
