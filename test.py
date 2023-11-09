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
from sklearn.cluster import KMeans, DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot
from torchvision.transforms import transforms

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


class ProtoNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProtoNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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

    def calculate_prototypes_DBSCAN(self, support_set, labels):
        """
        计算每个类别的聚类原型
        Args:
        support_set: 支持集样本, shape=(num_support, feature_dim)
        labels: 支持集样本的标签, shape=(num_support,)
        Returns:
        prototypes: 聚类原型, shape=(num_classes, feature_dim)
        """
        num_classes = torch.unique(labels).size(0)
        prototypes = torch.zeros(num_classes, support_set.size(1))
        for c in range(num_classes):
            class_support_set = support_set[labels == c]
            dbscan = DBSCAN(eps=0.5, min_samples=5)  # 设置 DBSCAN 的参数，可以根据具体数据进行调整
            dbscan.fit(class_support_set)
            unique_labels = np.unique(dbscan.labels_)
            num_clusters = len(unique_labels)
            class_prototype = torch.zeros(support_set.size(1))
            if num_clusters > 1:
                cluster_centers = []
                for i in range(num_clusters):
                    cluster_samples = class_support_set[dbscan.labels_ == i]
                    cluster_center = cluster_samples.mean(dim=0)
                    cluster_centers.append(cluster_center)
                cluster_centers = torch.stack(cluster_centers)
                distances = torch.cdist(class_support_set, cluster_centers)
                cluster_indices = torch.argmin(distances, dim=1)
                for i in range(num_clusters):
                    cluster_samples = class_support_set[cluster_indices == i]
                    prototype = cluster_samples.mean(dim=0)
                    class_prototype += prototype
                class_prototype /= num_clusters
            else:
                class_prototype = class_support_set.mean(dim=0)
            prototypes[c] = class_prototype
        return prototypes

    def calculate_prototypes_Kmeans(self, support_set, labels, num_clusters):  # 这个函数目前是正确的，不，是错误的，现在的问题是有可能会出现nan的值
        """
        计算每个类别的聚类原型
        Args:
            support_set: 支持集样本, shape=(num_support, feature_dim)
            labels: 支持集样本的标签, shape=(num_support,)
            num_clusters: 聚类的数量
        Returns:
            prototypes: 聚类原型, shape=(num_classes, num_clusters, feature_dim)
        """
        num_classes = torch.unique(labels).size(0)
        prototypes = torch.zeros(num_classes, num_clusters, support_set.size(1))
        for c in range(num_classes):  # 每一个类
            class_prototypes = torch.zeros(num_clusters, support_set.size(1))
            class_support_set = support_set[labels == c]
            kmeans = KMeans(n_clusters=num_clusters).fit(class_support_set)
            cluster_centers = kmeans.cluster_centers_
            for i in range(num_clusters):
                # cluster_samples = class_support_set[kmeans.labels_ == i]  #这是选择所有簇里面的样本取平均作为样本中心
                # prototype = cluster_samples.mean(dim=0)
                # class_prototypes[i] = prototype
                class_prototypes[i] = torch.from_numpy(cluster_centers[i])  # 直接选择样本中心
            # prototypes[c] = torch.from_numpy(class_prototypes)
            prototypes[c] = class_prototypes
        return prototypes

    def calculate_dynamic_prototype(self, query, prototypes, labels, num_clusters):
        """
        计算每个查询样本对应的动态原型
        Args:
            query: 查询集样本, shape=(num_query, feature_dim)
            prototypes: 聚类原型, shape=(num_classes, num_clusters, feature_dim)
            labels: 支持集样本的标签, shape=(num_support,)
            num_clusters: 聚类的数量
        Returns:
            dynamic_prototypes: 查询样本对应的动态原型, shape=(num_query, num_classes, feature_dim)
        """
        num_classes = prototypes.size(0)
        num_queries = query.size(0)
        distances = torch.zeros(num_queries, num_classes, num_clusters)
        for c in range(num_classes):
            # class_prototypes = prototypes[c]
            # class_query = query[labels == c]
            # kmeans = KMeans(n_clusters=num_clusters, init=class_prototypes).fit(class_query)
            class_distances = torch.cdist(query, prototypes[c])
            distances[:, c] = class_distances  # distance计算是错的

        # weights = 1 / distances   #这里使用的是归一化处理
        # weights_sum = weights.sum(dim=-1, keepdim=True)
        # weights_sum[weights_sum == 0] = 1e-8
        # weights_normalized = weights / weights_sum
        # dynamic_prototypes = torch.sum(weights_normalized.unsqueeze(1) * prototypes.unsqueeze(0), dim=2)
        weights = -distances  # 使用了softmax处理
        weights_softmax = torch.nn.functional.softmax(weights, dim=-1)
        # print(weights_softmax.shape)  # torch.Size([25, 5, 2])  num_query * num_class * cluster
        # print(prototypes.shape)  # torch.Size([5, 2, 16])    num_class * cluster * feature_dim
        # dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(1) * prototypes.unsqueeze(0), dim=2)
        dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(-1) * prototypes.unsqueeze(0).unsqueeze(0), dim=-2
                                       )
        # unsqueeze的用法是在指定维度增加1，dim=0表示在第一个维度上增加维度，dim等于1表示在第二个维度上增加1
        return dynamic_prototypes

    # def predict(self, query_set, support_set, support_labels):   #这个是直接得到对应的标签
    #     prototypes = self.compute_prototypes(support_set, support_labels)
    #     distances = torch.cdist(query_set, prototypes)
    #     predictions = torch.argmin(distances, dim=1)
    #     return predictions

    def predict_DBSCAN(self, query_set, support_set, support_labels):   #这个可以求得属于每个类别的概率，可以用来做集成学习
        # prototypes = self.compute_prototypes(support_set, support_labels)
        prototypes = self.calculate_prototypes_DBSCAN(support_set, support_labels)
        distances = torch.cdist(query_set, prototypes)
        logits = -distances  # negative distances to use softmax instead of argmin
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def predict(self, query_set, support_set, support_labels):   #这个可以求得属于每个类别的概率，可以用来做集成学习
        prototypes = self.compute_prototypes(support_set, support_labels)
        # prototypes = self.calculate_prototypes_DBSCAN(support_set, support_labels)
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



def evaluate(model, criterion, dataset, num_classes, num_support, num_query, num_tasks):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_accuracy_dynamic = 0.0    #K-means聚类
    total_accuracy_DNSCAN = 0.0
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

                # 计算动态原型,使用每个簇的样本中心，得到的维度是num_query * num_classes * feature_dim
                prototypes_per_dynamic = model.calculate_prototypes_Kmeans(support_embeddings, support_labels, 2)  #测试动态原型
                dynamic_prototypes = model.calculate_dynamic_prototype(query_embeddings, prototypes_per_dynamic, support_labels, 2).squeeze()  #去掉1维度
                # print(dynamic_prototypes.shape)
                prototypes = model.compute_prototypes(support_embeddings, support_labels)  #普通的原型计算，得到的维度是num_classes * feature_dim


                #然后计算的距离维度应该是num_query * num_classes
                distances = torch.cdist(query_embeddings, prototypes)

                distances_dynamic = []
                for i in range(len(query_embeddings)):
                    # 对于查询样本 i，计算它与所有类别之间的距离
                    query_embedding_i = query_embeddings[i:i + 1]
                    prototypes_i = dynamic_prototypes[i]
                    # print(query_embedding_i.shape)
                    # print(prototypes_i.shape)
                    distance_i = torch.cdist(query_embedding_i, prototypes_i, p=2.0)
                    distances_dynamic.append(distance_i)
                # 将距离列表转换为张量
                distances_dynamic = torch.cat(distances_dynamic, dim=0)

                # distances_dynamic = torch.cdist(query_embeddings, dynamic_prototypes)
                # print(distances.shape)
                # print(distances_dynamic.shape)

                loss = criterion(distances, query_labels)  #这两句话会导致模型出错
                total_loss += loss.item()


                probabilities_dynamic = F.softmax(-distances_dynamic, dim=1)
                predictions_dynamic = torch.argmax(probabilities_dynamic, dim=1)
                accuracy_dynamic = (predictions_dynamic == query_labels).float().mean()
                total_accuracy_dynamic += accuracy_dynamic.item()

                probabilities_DNSCAN = model.predict(query_embeddings, support_embeddings, support_labels)
                predictions_DNSCAN = torch.argmax(probabilities_DNSCAN, dim=1)
                accuracy_DNSCAN = (predictions_DNSCAN == query_labels).float().mean()
                total_accuracy_DNSCAN += accuracy_DNSCAN.item()

                probabilities = model.predict(query_embeddings, support_embeddings, support_labels)
                predictions = torch.argmax(probabilities, dim=1)
                # predictions = model.predict(query_embeddings, support_embeddings, support_labels)
                accuracy = (predictions == query_labels).float().mean()
                total_accuracy += accuracy.item()

                print("Task_{} accuracy: {}, dynamic_accuracy: {}, DNSCAN_accuracy: {}".format(i, accuracy, accuracy_dynamic, accuracy_DNSCAN))
            except Exception as e:
                print("Task_test {} failed: {}".format(i, e))
                continue
    return total_loss / num_tasks, total_accuracy / num_tasks, total_accuracy_dynamic/num_tasks, total_accuracy_DNSCAN/num_tasks

#这里是测试了K-means的计算动态原型，发现其实如果不是有的cluster会比较少，导致全是nan值，其实还好。比如我们可以选择cluster为3
if __name__ == '__main__':
    # Hyperparameters
    input_size = 57
    hidden_size = 114
    output_size = 16
    lr = 1e-3
    epochs = 10
    batch_size = 32
    num_classes = 5
    num_support = 8
    num_query = 2
    num_tasks = 50
    # num_tasks = 100

    # Load data


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
        val_loss, val_acc, val_acc_dynamic, val_acc_DASCAN = evaluate(model, criterion, dataset, num_classes, num_support, num_query, num_tasks)
        print(f'Epoch {epoch + 1}:  Val Loss={val_loss:.4f} Val Acc={val_acc:.4f} Val Acc dynamic={val_acc_dynamic:.4f} Val acc DBSCAN={val_acc_DASCAN:.4f}')
        # print(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}')
