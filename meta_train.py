import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import numpy as np
from dataset import CustomDataset # 自定义的数据集类
# from prototypical_network import ProtoNet # 自定义的Protonet模型类
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms


class ProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProtoNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )
        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 16)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def compute_prototypes(self, support_set, labels):
        unique_labels = torch.unique(labels)
        # prototypes = torch.zeros(len(unique_labels), self.hidden_size).to(support_set.device)
        prototypes = torch.zeros(len(unique_labels), self.output_size).to(support_set.device)
        for i, label in enumerate(unique_labels):
            prototypes[i] = support_set[labels == label].mean(dim=0)

        return prototypes

    # def predict(self, query_set, support_set, support_labels):   #这个是直接得到对应的标签
    #     prototypes = self.compute_prototypes(support_set, support_labels)
    #     distances = torch.cdist(query_set, prototypes)
    #     predictions = torch.argmin(distances, dim=1)
    #     return predictions

    def predict(self, query_set, support_set, support_labels):   #这个可以求得属于每个类别的概率，可以用来做集成学习
        prototypes = self.compute_prototypes(support_set, support_labels)
        distances = torch.cdist(query_set, prototypes)
        logits = -distances  # negative distances to use softmax instead of argmin
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def generate_task(dataset, num_classes, num_support, num_query):

    classes = torch.randperm(len(dataset.labels))[:num_classes+5]

    support_set = []
    query_set = []
    support_labels = []
    query_labels = []
    j = 0  #表示query的标签，每当搜寻到一个符合的类之后就加1
    for i, c in enumerate(classes):
        # 获取该类别的所有样本
        samples = [idx for idx, (_, label) in enumerate(dataset) if label == c]
        if(len(samples) < (num_query + num_support)):  #如果样本数量不足，跳过该类
            # i = i - 1
            continue
        # 随机选择 num_support + num_query 个样本
        samples = torch.tensor(samples)[torch.randperm(len(samples))][:num_support + num_query]
        support_set.append(dataset[samples[:num_support]][0])
        query_set.append(dataset[samples[num_support:]][0])
        support_labels.append(torch.ones(num_support, dtype=torch.long) * j)
        query_labels.append(torch.ones(num_query, dtype=torch.long) * j)
        j = j + 1
        if(len(query_set) == num_classes):   #获取到足够多的类，就跳出
            break


    #拼接为一个长向量
    # support_set = torch.stack(support_set).reshape(-1, *support_set[0][0].shape)
    # query_set = torch.stack(query_set).reshape(-1, *query_set[0][0].shape)
    # print(set([x.shape for x in support_set]))
    # print(set([x.shape for x in query_set]))
    support_set = torch.tensor(np.stack(support_set)).reshape(-1, *support_set[0][0].shape)
    query_set = torch.tensor(np.stack(query_set)).reshape(-1, *query_set[0][0].shape)
    support_labels = torch.cat(support_labels)
    query_labels = torch.cat(query_labels)

    perm = torch.randperm(len(support_set))
    support_set = support_set[perm]
    support_labels = support_labels[perm]

    perm = torch.randperm(len(query_set))
    query_set = query_set[perm]
    query_labels = query_labels[perm]

    return support_set, query_set, support_labels, query_labels

def train(model, optimizer, criterion, dataset, num_classes, num_support, num_query, num_tasks):
    model.train()
    total_loss = 0.0

    for i in range(num_tasks):
        try:
            support_set, query_set, support_labels, query_labels = generate_task(dataset, num_classes, num_support,
                                                                                 num_query)
            optimizer.zero_grad()
            support_set = support_set.view(-1, support_set.shape[-1])
            query_set = query_set.view(-1, query_set.shape[-1])
            support_labels = support_labels.view(-1)
            query_labels = query_labels.view(-1)

            support_set = support_set.to(device)
            query_set = query_set.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

            support_embeddings = model(support_set)
            query_embeddings = model(query_set)

            prototypes = model.compute_prototypes(support_embeddings, support_labels)
            distances = torch.cdist(query_embeddings, prototypes)

            loss = criterion(distances, query_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        except Exception as e:
            print("Task {} failed: {}".format(i, e))
            continue
    return total_loss / num_tasks


def evaluate(model, criterion, dataset, num_classes, num_support, num_query, num_tasks):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for i in range(num_tasks):
            try:
                support_set, query_set, support_labels, query_labels = generate_task(dataset, num_classes, num_support,
                                                                                     num_query)
                support_set = support_set.view(-1, support_set.shape[-1])
                query_set = query_set.view(-1, query_set.shape[-1])
                support_labels = support_labels.view(-1)
                query_labels = query_labels.view(-1)

                support_set = support_set.to(device)
                query_set = query_set.to(device)
                support_labels = support_labels.to(device)
                query_labels = query_labels.to(device)

                support_embeddings = model(support_set)
                query_embeddings = model(query_set)

                prototypes = model.compute_prototypes(support_embeddings, support_labels)
                distances = torch.cdist(query_embeddings, prototypes)

                loss = criterion(distances, query_labels)  #这两句话会导致模型出错
                total_loss += loss.item()

                probabilities = model.predict(query_embeddings, support_embeddings, support_labels)
                predictions = torch.argmax(probabilities, dim=1)
                # predictions = model.predict(query_embeddings, support_embeddings, support_labels)
                accuracy = (predictions == query_labels).float().mean()
                total_accuracy += accuracy.item()
            except Exception as e:
                print("Task_test {} failed: {}".format(i, e))
                continue
    return total_loss / num_tasks, total_accuracy / num_tasks


if __name__ == '__main__':
    # Hyperparameters
    input_size = 57
    hidden_size = 114
    output_size = 16
    lr = 1e-3
    epochs = 10
    batch_size = 32
    num_classes = 5
    num_support = 5
    num_query = 5
    num_tasks = 5
    # num_tasks = 100

    # Load data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # train_dataset = Omniglot(root='./data', background=True, transform=transform, download=True)
    # val_dataset = Omniglot(root='./data', background=False, transform=transform, download=True)
    dataset = CustomDataset('./dataset/Test/CIC2019_per.csv')
    # dataset = CustomDataset('./dataset/Test/CIC2019_opcode.csv')
    # train_size = int(len(dataset) * 0.8)
    # test_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  #这里不能这么用
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  #换成CPU就不会报错了hh
    #还有很多地方没改，包括嵌入网络结构，然后载入已经预训练过的参数。然后尝试一下这个可不可行吧

    # Initialize model and optimizer
    # model = ProtoNet(input_size, hidden_size, output_size).to(device)
    model = ProtoNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load('./model/Siamese_per.pth'))  #加载模型预训练参数
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    for epoch in range(epochs):
        # train_loss = train(model, optimizer, criterion, dataset, num_classes, num_support, num_query, num_tasks)
        val_loss, val_acc = evaluate(model, criterion, dataset, num_classes, num_support, num_query, num_tasks)
        print(f'Epoch {epoch + 1}:  Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}')
        # print(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}')
