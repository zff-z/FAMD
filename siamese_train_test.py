import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SiameseDataset
from ensemble_learning import run



class SiameseNet(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize):
        super(SiameseNet, self).__init__()
        # self.fc1 = nn.Linear(343, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 16)
        self.fc1 = nn.Linear(inputsize, hiddensize)
        self.fc2 = nn.Linear(hiddensize, hiddensize)
        self.fc3 = nn.Linear(hiddensize, outputsize)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance1 = F.pairwise_distance(anchor, positive, 2)
        distance2 = F.pairwise_distance(anchor, negative, 2)
        loss = torch.mean((distance1 - distance2 + self.margin) ** 2)
        return loss

inputsize = 478
hiddensize = 200
outputsize = 16
model = SiameseNet(inputsize, hiddensize, outputsize)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = SiameseDataset('./dataset/Train/Drebin_per_API_opcode.csv')
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
skip_epoch = 8
for epoch in range(20):
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        optimizer.zero_grad()
        output1 = model(anchor)
        output2 = model(positive)
        output3 = model(negative)
        loss = criterion(output1, output2, output3)
        loss.backward()
        optimizer.step()

    print('Epoch: {} Loss: {:.6f}'.format(epoch + 1, loss.item()))


    if epoch > skip_epoch:
        run(model)




