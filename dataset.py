import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

class SiameseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.data['family'] = self.label_encoder.fit_transform(self.data['family'])
        self.labels = list(set(self.data['family']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample1 = self.data.iloc[idx, 1:-1].values.astype('float32')
        sample1 = self.data.iloc[idx, 1:-1].values.astype('float32')
        label1 = self.data.iloc[idx, -1]
        # 随机选择不同的样本作为正样本
        while True:
            idx2 = np.random.randint(0, len(self.data))
            if self.data.iloc[idx2, -1] == label1:
                break
        # sample2 = self.data.iloc[idx2, 1:-1].values.astype('float32')
        sample2 = self.data.iloc[idx2, 1:-1].values.astype('float32')
        label2 = self.data.iloc[idx2, -1]
        # 随机选择不同的标签作为负样本
        while True:
            label3 = np.random.choice(self.labels)
            if label3 != label1:
                break
        idx3 = np.random.choice(self.data[self.data['family'] == label3].index)
        # sample3 = self.data.iloc[idx3, 1:-1].values.astype('float32')
        sample3 = self.data.iloc[idx3, 1:-1].values.astype('float32')
        return sample1, sample2, sample3

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.data['family'] = self.label_encoder.fit_transform(self.data['family'])
        self.labels = list(set(self.data['family']))
        self.hashes = list(set(self.data['hash']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取数据
        features = self.data.iloc[idx, 1:-1].values.astype('float32')  # 去除最后一列label，其他的是feature
        # 获取标签并进行编码
        label = self.data.iloc[idx, -1]

        #hash值获取
        hash = self.data.iloc[idx, 0]

        return features, label, hash



class Per_API_Opcode_Dataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.data['family'] = self.label_encoder.fit_transform(self.data['family'])
        self.labels = list(set(self.data['family']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取数据
        per_features = self.data.iloc[idx, 1:58].values.astype('float32')  # 去除最后一列label，其他的是feature
        API_features = self.data.iloc[idx, 58:136].values.astype('float32')  # 去除最后一列label，其他的是feature
        opcode_features = self.data.iloc[idx, 136:-1].values.astype('float32')
        # 获取标签并进行编码
        label = self.data.iloc[idx, -1]

        return (per_features, API_features, opcode_features), label



