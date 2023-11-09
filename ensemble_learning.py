import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from dataset import Per_API_Opcode_Dataset, CustomDataset
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def generate_task(dataset, num_classes, num_support, num_query):

    classes = torch.randperm(len(dataset.labels))[:num_classes+10]

    support_set = []
    query_set = []
    support_labels = []
    query_labels = []
    query_hashes = []
    j = 0
    #这里目前还有点问题，没搞定
    for i, c in enumerate(classes):
        # 获取该类别的所有样本
        samples = [idx for idx, (_, label, _) in enumerate(dataset) if label == c]
        if(len(samples) < (num_query + num_support)):  #如果样本数量不足，跳过该类
            # i = i - 1
            continue
        # 随机选择 num_support + num_query 个样本
        samples = torch.tensor(samples)[torch.randperm(len(samples))][:num_support + num_query]
        support_set.append(dataset[samples[:num_support]][0])
        query_set.append(dataset[samples[num_support:]][0])
        support_labels.append(torch.ones(num_support, dtype=torch.long) * j)
        query_labels.append(torch.ones(num_query, dtype=torch.long) * j)

        # 存储查询样本的哈希值
        query_hashes.extend([dataset.hashes[idx] for idx in samples[num_support:]])

        j = j + 1
        if(len(query_set) == num_classes):   #获取到足够多的类，就跳出
            break


    #拼接为一个长向量
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
    query_hashes = query_hashes[perm]

    return support_set, query_set, support_labels, query_labels, query_hashes
# 定义分类器模型

import numpy as np

def calculate_weights(predictions, method='entropy', ):
    """
    Calculate the weights of each model based on their predictions.

    Args:
        predictions: A list of PyTorch tensors, where each tensor contains the predicted probabilities of each class for a single model.
        method: The method used to calculate weights. Currently, only 'entropy' method is supported.

    Returns:
        A numpy array of weights for each model, with the same shape as the input predictions.
    """
    num_models = len(predictions)
    num_samples = predictions[0].shape[0]
    weights = torch.zeros((num_samples, num_models))

    if method == 'entropy':
        # Calculate the entropy of each model's predictions for each sample
        entropy = torch.zeros((num_samples, num_models))
        for i in range(num_models):
            entropy[:, i] = -torch.sum(predictions[i] * torch.log(predictions[i]), dim=1)

        # Calculate the weights based on the entropy for each sample
        weights = torch.exp(-entropy)
        weights /= torch.sum(weights, dim=1, keepdim=True)

    return weights.numpy()

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

    def compute_prototypes(self, support_set, labels):                #计算方式:直接取类中样本的平均值
        unique_labels = torch.unique(labels)
        prototypes = torch.zeros(len(unique_labels), self.output_size).to(support_set.device)

        for i, label in enumerate(unique_labels):
            prototypes[i] = support_set[labels == label].mean(dim=0)

        return prototypes

    # def compute_prototypes(self, support_set, labels, num_clusters, query):   #计算方式：通过K-Means聚类得到多个原型，然后通过注意力机制实现动态原型
    #     # Compute prototypes for each cluster in each class
    #     print(support_set.shape)
    #     print(labels.shape)
    #     print(query.shape)
    #     unique_labels = torch.unique(labels)
    #     prototypes = []
    #     for label in unique_labels:
    #         # Get support samples for this class
    #         class_samples = support_set[labels == label]
    #
    #         # Cluster support samples into num_clusters clusters
    #         kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(class_samples)
    #         cluster_centers = kmeans.cluster_centers_
    #
    #         # Compute prototypes for each cluster
    #         # for center in cluster_centers:
    #         for i in range(num_clusters):
    #             # prototype = center.mean(dim=0)
    #             cluster_samples = class_samples[kmeans.labels_ == i]
    #             prototype = cluster_samples.mean(dim=0)
    #             prototypes.append(prototype)
    #
    #     # Compute weighted average of prototypes based on distance to query
    #     distances = torch.cdist(query.unsqueeze(0), torch.stack(prototypes))
    #     weights = 1 / distances
    #     weighted_prototypes = weights * torch.stack(prototypes)
    #     final_prototype = weighted_prototypes.sum(dim=0) / weights.sum()
    #
    #     return final_prototype

    # def compute_prototypes(self, support_set, labels, query):  #计算方式：通过DBSCAN聚类得到多个原型，然后通过注意力机制实现动态原型
    #     # Compute prototypes for each cluster in each class using DBSCAN
    #     unique_labels = torch.unique(labels)
    #     prototypes = []
    #     for label in unique_labels:
    #         # Get support samples for this class
    #         class_samples = support_set[labels == label]
    #
    #         # Cluster support samples using DBSCAN
    #         dbscan = DBSCAN(eps=0.5, min_samples=5).fit(class_samples)
    #         cluster_centers = []
    #         for cluster_label in torch.unique(dbscan.labels_):
    #             if cluster_label == -1:
    #                 continue
    #             cluster_samples = class_samples[dbscan.labels_ == cluster_label]
    #             cluster_center = cluster_samples.mean(dim=0)
    #             cluster_centers.append(cluster_center)
    #
    #         # Compute prototypes for each cluster
    #         for center in cluster_centers:
    #             prototype = center.mean(dim=0)
    #             prototypes.append(prototype)
    #
    #     # Compute weighted average of prototypes based on distance to query
    #     distances = torch.cdist(query.unsqueeze(0), torch.stack(prototypes))
    #     weights = 1 / distances
    #     weighted_prototypes = weights * torch.stack(prototypes)
    #     final_prototype = weighted_prototypes.sum(dim=0) / weights.sum()
    #
    #     return final_prototype
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
        dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(-1) * prototypes.unsqueeze(0).unsqueeze(0), dim=-2)
        # unsqueeze的用法是在指定维度增加1，dim=0表示在第一个维度上增加维度，dim等于1表示在第二个维度上增加1
        return dynamic_prototypes

    def predict(self, query_set, support_set, support_labels):
        prototypes = self.compute_prototypes(support_set, support_labels)
        distances = torch.cdist(query_set, prototypes)
        logits = -distances  # negative distances to use softmax instead of argmin
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def evaluate(model, criterion, dataset, num_classes, num_support, num_query, num_tasks):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for i in range(num_tasks):
            support_set, query_set, support_labels, query_labels, query_hashes = generate_task(dataset, num_classes, num_support,
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

            # predictions = model.predict(query_embeddings, support_embeddings, support_labels)
            probabilities = model.predict(query_embeddings, support_embeddings, support_labels)
            predictions = torch.argmax(probabilities, dim=1)
            # predictions = []
            # for i in range(len(query_labels)):
            #     max_prob, max_index = torch.max(probabilities[i], dim=0)
            #     predicted_label = support_labels[max_index]
            #     predictions.append(predicted_label)

            accuracy = (predictions == query_labels).float().mean()
            total_accuracy += accuracy.item()

    return total_loss / num_tasks, total_accuracy / num_tasks

def All_test(model_per, model_API, model_opcode, criterion, dataset, num_classes, num_support, num_query, num_tasks, num_cluster):
    model_per.eval()
    model_API.eval()
    model_opcode.eval()

    per_loss = 0.0
    API_loss = 0.0
    opcode_loss = 0.0
    total_loss = 0.0
    total_accuracy = 0.0
    total_accuracy_per = 0.0
    total_accuracy_API = 0.0
    total_accuracy_opcode = 0.0
    total_accuracy_per_dynamic = 0.0
    total_accuracy_API_dynamic = 0.0
    total_accuracy_opcode_dynamic = 0.0
    total_accuracy_dynamic = 0.0
    total_loss_per = 0.0
    total_loss_API = 0.0
    total_loss_opcode = 0.0

    with torch.no_grad():
        for i in range(num_tasks):
            try:
                support_set, query_set, support_labels, query_labels, query_hashes = generate_task(dataset, num_classes, num_support,
                                                                                     num_query)
                # print(support_set.shape)
                support_set = support_set.view(-1, support_set.shape[-1])
                # print(support_set.shape)
                query_set = query_set.view(-1, query_set.shape[-1])
                support_labels = support_labels.view(-1)
                query_labels = query_labels.view(-1)
                #切分为三个特征子集
                support_set_per = support_set[:, :57]
                support_set_API = support_set[:, 57:135]
                support_set_opcode = support_set[:, 135:]

                query_set_per = query_set[:, :57]
                query_set_API = query_set[:, 57:135]
                query_set_opcode = query_set[:, 135:]

                #把hash值也加进来了

                # support_set_per = support_set[:, 1:58]
                # support_set_API = support_set[:, 58:136]
                # support_set_opcode = support_set[:, 136:]
                #
                # query_set_hash = query_set[:, :1]
                # query_set_per = query_set[:, 1:58]
                # query_set_API = query_set[:, 58:136]
                # query_set_opcode = query_set[:, 136:]

                #数据加载进GPU
                support_set_per = support_set_per.to(device)
                support_set_API = support_set_API.to(device)
                support_set_opcode = support_set_opcode.to(device)
                query_set_per = query_set_per.to(device)
                query_set_API = query_set_API.to(device)
                query_set_opcode = query_set_opcode.to(device)

                # support_set = support_set.to(device)
                # query_set = query_set.to(device)
                # support_labels = support_labels.to(device)
                # query_labels = query_labels.to(device)

                #输入模型得到嵌入向量
                support_per_embeddings = model_per(support_set_per)
                support_API_embeddings = model_API(support_set_API)
                support_opcode_embeddings = model_opcode(support_set_opcode)
                query_per_embeddings = model_per(query_set_per)
                query_API_embeddings = model_API(query_set_API)
                query_opcode_embeddings = model_opcode(query_set_opcode)
                # support_embeddings = model(support_set)
                # query_embeddings = model(query_set)

                #计算原型，并计算距离
                prototypes_per_dynamic = model_per.calculate_prototypes_Kmeans(support_per_embeddings, support_labels, num_cluster)  #测试动态原型
                dynamic_prototypes_per = model_per.calculate_dynamic_prototype(query_per_embeddings, prototypes_per_dynamic, support_labels, num_cluster).squeeze()
                # print("动态原型的维度：{}".format(dynamic_prototypes_per.shape))
                prototypes_API_dynamic = model_API.calculate_prototypes_Kmeans(support_API_embeddings, support_labels,num_cluster)  # 测试动态原型
                dynamic_prototypes_API = model_API.calculate_dynamic_prototype(query_API_embeddings, prototypes_API_dynamic,support_labels, num_cluster).squeeze()
                prototypes_opcode_dynamic = model_opcode.calculate_prototypes_Kmeans(support_opcode_embeddings, support_labels,num_cluster)  # 测试动态原型
                dynamic_prototypes_opcode = model_opcode.calculate_dynamic_prototype(query_opcode_embeddings, prototypes_opcode_dynamic,support_labels, num_cluster).squeeze()

                #这里是最初的原型计算方法
                prototypes_per = model_per.compute_prototypes(support_per_embeddings, support_labels)
                prototypes_API = model_API.compute_prototypes(support_API_embeddings, support_labels)
                prototypes_opcode = model_opcode.compute_prototypes(support_opcode_embeddings, support_labels)

                #下面是动态的距离计算
                distances_dynamic_per = []
                for i in range(len(query_per_embeddings)):
                    # 对于查询样本 i，计算它与所有类别之间的距离
                    query_embedding_i = query_per_embeddings[i:i + 1]
                    prototypes_i = dynamic_prototypes_per[i]
                    distance_i = torch.cdist(query_embedding_i, prototypes_i, p=2.0)
                    distances_dynamic_per.append(distance_i)
                # 将距离列表转换为张量
                distances_dynamic_per = torch.cat(distances_dynamic_per, dim=0)

                distances_dynamic_API = []
                for i in range(len(query_API_embeddings)):
                    # 对于查询样本 i，计算它与所有类别之间的距离
                    query_embedding_i = query_API_embeddings[i:i + 1]
                    prototypes_i = dynamic_prototypes_API[i]
                    distance_i = torch.cdist(query_embedding_i, prototypes_i, p=2.0)
                    distances_dynamic_API.append(distance_i)
                # 将距离列表转换为张量
                distances_dynamic_API = torch.cat(distances_dynamic_API, dim=0)

                distances_dynamic_opcode = []
                for i in range(len(query_opcode_embeddings)):
                    # 对于查询样本 i，计算它与所有类别之间的距离
                    query_embedding_i = query_opcode_embeddings[i:i + 1]
                    prototypes_i = dynamic_prototypes_opcode[i]
                    distance_i = torch.cdist(query_embedding_i, prototypes_i, p=2.0)
                    distances_dynamic_opcode.append(distance_i)
                # 将距离列表转换为张量
                distances_dynamic_opcode = torch.cat(distances_dynamic_opcode, dim=0)


                distances_per = torch.cdist(query_per_embeddings, prototypes_per)
                # distances_per = torch.cdist(query_per_embeddings, dynamic_prototypes)
                #这里的distance_per应该也和distance_API一样，只不过不同的query是和不同的prototype计算而已。就是要怎么分清这个维度的变化，有点头疼
                #还有就是原来的代码里有大量的重复计算，计算出来了distance还计算了predictions里又计算了一次
                #最后一个问题是动态的prototype虽然维度是对的，但是我不敢保证这个结果真的是对的，需要验证一下
                distances_API = torch.cdist(query_API_embeddings, prototypes_API)
                distances_opcode = torch.cdist(query_opcode_embeddings, prototypes_opcode)

                #使用动态原型预测的准确率
                scores_dynamic_per = F.softmax(-distances_dynamic_per, dim=-1)
                predicted_labels_dynamic_per = scores_dynamic_per.argmax(dim=-1)
                accuracy_dynamic_per = (predicted_labels_dynamic_per == query_labels).float().mean()

                scores_dynamic_API = F.softmax(-distances_dynamic_API, dim=-1)
                predicted_labels_dynamic_API = scores_dynamic_API.argmax(dim=-1)
                accuracy_dynamic_API = (predicted_labels_dynamic_API == query_labels).float().mean()

                scores_dynamic_opcode = F.softmax(-distances_dynamic_opcode, dim=-1)
                predicted_labels_dynamic_opcode = scores_dynamic_opcode.argmax(dim=-1)
                accuracy_dynamic_opcode = (predicted_labels_dynamic_opcode == query_labels).float().mean()

                probabilities_all_dynamic = [scores_dynamic_per, scores_dynamic_API, scores_dynamic_opcode]
                weights_dynamic = calculate_weights(probabilities_all_dynamic)


                weighted_probabilities_per_dynamic = scores_dynamic_per * weights_dynamic[:, 0].reshape(-1, 1)
                weighted_probabilities_API_dynamic = scores_dynamic_API * weights_dynamic[:, 1].reshape(-1, 1)
                weighted_probabilities_opcode_dynamic = scores_dynamic_opcode * weights_dynamic[:, 2].reshape(-1, 1)
                probabilities_dynamic = weighted_probabilities_per_dynamic + weighted_probabilities_API_dynamic + weighted_probabilities_opcode_dynamic
                predictions_dynamic = torch.argmax(probabilities_dynamic, dim=1)
                accuracy_dynamic = (predictions_dynamic == query_labels).float().mean()
                 #统计预测出错的样本的index
                incorrect_indexes = torch.nonzero(predictions_dynamic != query_labels).view(-1)

                # 找到预测出错的哈希值
                incorrect_hashes = [query_hashes[idx] for idx in incorrect_indexes]
                print(incorrect_hashes)

                total_accuracy_per_dynamic += accuracy_dynamic_per.item()
                total_accuracy_API_dynamic += accuracy_dynamic_API.item()
                total_accuracy_opcode_dynamic += accuracy_dynamic_opcode.item()
                total_accuracy_dynamic += accuracy_dynamic.item()

                scores_per = F.softmax(-distances_per, dim=-1)
                predicted_labels_per = scores_per.argmax(dim=-1)
                accuracy_per = (predicted_labels_per == query_labels).float().mean()
                scores_API = F.softmax(-distances_API, dim=-1)
                predicted_labels_API = scores_API.argmax(dim=-1)
                accuracy_API = (predicted_labels_API == query_labels).float().mean()
                scores_opcode = F.softmax(-distances_opcode, dim=-1)
                predicted_labels_opcode = scores_opcode.argmax(dim=-1)
                accuracy_opcode = (predicted_labels_opcode == query_labels).float().mean()
                #计算损失
                loss_per = criterion(distances_per, query_labels)
                total_loss_per += loss_per.item()
                loss_API = criterion(distances_API, query_labels)
                total_loss_API += loss_API.item()
                loss_opcode = criterion(distances_opcode, query_labels)
                total_loss_opcode += loss_opcode.item()
                total_loss = (total_loss_per + total_loss_API + total_loss_opcode) / 3
                # loss = criterion(distances, query_labels)  #这两句话会导致模型出错
                # total_loss += loss.item()

                probabilities_per = F.softmax(-distances_per, dim=1)
                probabilities_API = F.softmax(-distances_API, dim=1)
                probabilities_opcode = F.softmax(-distances_opcode, dim=1)
                probabilities_all = [probabilities_per, probabilities_API, probabilities_opcode]
                weights = calculate_weights(probabilities_all)

                # 将三个模型的预测输出概率分别与对应的权重相乘

                weighted_probabilities_per = probabilities_per * weights[:, 0].reshape(-1, 1)
                weighted_probabilities_API = probabilities_API * weights[:, 1].reshape(-1, 1)
                weighted_probabilities_opcode = probabilities_opcode * weights[:, 2].reshape(-1, 1)
                probabilities = weighted_probabilities_per + weighted_probabilities_API + weighted_probabilities_opcode
                predictions = torch.argmax(probabilities, dim=1)

                total_accuracy_per += accuracy_per.item()
                total_accuracy_API += accuracy_API.item()
                total_accuracy_opcode += accuracy_opcode.item()
                accuracy = (predictions == query_labels).float().mean()
                total_accuracy += accuracy.item()

                # probabilities_per = model_per.predict(query_per_embeddings, support_per_embeddings, support_labels)
                # probabilities_API = model_API.predict(query_API_embeddings, support_API_embeddings, support_labels)
                # probabilities_opcode = model_opcode.predict(query_opcode_embeddings, support_opcode_embeddings, support_labels)
                # predictions_per = torch.argmax(probabilities_per, dim=1)
                # predictions_API = torch.argmax(probabilities_API, dim=1)
                # predictions_opcode = torch.argmax(probabilities_opcode, dim=1)

                # probabilities_all = [probabilities_per, probabilities_API, probabilities_opcode]
                # weights = calculate_weights(probabilities_all)


                # 将三个模型的预测输出概率分别与对应的权重相乘

                # weighted_probabilities_per = probabilities_per * weights[:, 0].reshape(-1, 1)
                # weighted_probabilities_API = probabilities_API * weights[:, 1].reshape(-1, 1)
                # weighted_probabilities_opcode = probabilities_opcode * weights[:, 2].reshape(-1, 1)
                # probabilities = weighted_probabilities_per + weighted_probabilities_API + weighted_probabilities_opcode
                # predictions = torch.argmax(probabilities, dim=1)


                # accuracy_per = (predictions_per == query_labels).float().mean()
                # accuracy_API = (predictions_API == query_labels).float().mean()
                # accuracy_opcode = (predictions_opcode == query_labels).float().mean()
                # total_accuracy_per += accuracy_per.item()
                # total_accuracy_API += accuracy_API.item()
                # total_accuracy_opcode += accuracy_opcode.item()
                #
                # accuracy = (predictions == query_labels).float().mean()
                # total_accuracy += accuracy.item()
            except Exception as e:
                print("task" + str(i) + "异常",  e)

    return total_loss / num_tasks, total_accuracy_per / num_tasks, total_accuracy_API / num_tasks, total_accuracy_opcode / num_tasks, total_accuracy / num_tasks, total_accuracy_per_dynamic / num_tasks,\
           total_accuracy_API_dynamic / num_tasks, total_accuracy_opcode_dynamic / num_tasks, total_accuracy_dynamic / num_tasks

if __name__ == '__main__':
    # Hyperparameters
    input_size = 334
    hidden_size = 16
    output_size = 10
    lr = 1e-3
    epochs = 3
    batch_size = 32
    num_classes = 5
    num_support = 5
    num_query = 4
    num_tasks = 100
    num_cluster = 3

    # optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # dataset = CustomDataset('./dataset/Test/obfuscation/PCM_per_API_opcode_new_label_common.csv')
    dataset = CustomDataset('/home/zf/siamese_chatGPT/dataset/Test/few-shot_drebin_9-20_sample.csv')

    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_per = ProtoNet(57, 114, 16).to(device)
    model_per.load_state_dict(torch.load('./model/model_right/Siamese_per.pth'))
    model_per.to(device)

    model_API = ProtoNet(78, 156, 16).to(device)
    model_API.load_state_dict(torch.load('./model/model_right/Siamese_API.pth'))
    model_API.to(device)

    model_opcode = ProtoNet(343, 200, 16).to(device)
    model_opcode.load_state_dict(torch.load('./model/model_right/Siamese_opcode.pth'))
    model_opcode.to(device)

#思路：让每个找到自己的概率是多大，然后综合在一起，先看一下每个单独的模型是否可以正常输出概率呢

    accuracy_per = []
    accuracy_API = []
    accuracy_opcode = []
    accuracy_ensembel = []
    accuracy_per_dynamic = []
    accuracy_API_dynamic = []
    accuracy_opcode_dynamic = []
    accuracy_ensembel_dynamic = []
    # Train and evaluate
    for epoch in range(epochs):
        # train_loss = train(model, optimizer, criterion, dataset, num_classes, num_support, num_query, num_tasks)
        val_loss, val_acc_per, val_acc_API, val_acc_opcode, val_acc, val_acc_per_dynamic, val_acc_API_dynamic, val_acc_opcode_dynamic, val_acc_dynamic = All_test(model_per, model_API, model_opcode, criterion, dataset, num_classes, num_support, num_query, num_tasks, num_cluster)
        print(f'Epoch {epoch + 1}:  Val Loss={val_loss:.4f} Val Acc per={val_acc_per:.4f} Val Acc API={val_acc_API:.4f} Val Acc opcode={val_acc_opcode:.4f} Val Acc={val_acc:.4f}')
        print(f'Val Acc per dynamic={val_acc_per_dynamic:.4f} Val Acc API dynamic={val_acc_API_dynamic:.4f} Val Acc opcode dynamic={val_acc_opcode_dynamic:.4f} Val Acc dynamic={val_acc_dynamic:.4f}')
        accuracy_per.append(round(val_acc_per, 4))
        accuracy_API.append(round(val_acc_API, 4))
        accuracy_opcode.append(round(val_acc_opcode, 4))
        accuracy_ensembel.append(round(val_acc, 4))
        accuracy_per_dynamic.append(round(val_acc_per_dynamic, 4))
        accuracy_API_dynamic.append(round(val_acc_API_dynamic, 4))
        accuracy_opcode_dynamic.append(round(val_acc_opcode_dynamic, 4))
        accuracy_ensembel_dynamic.append(round(val_acc_dynamic, 4))
        # print(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}')
    print(accuracy_per)
    print(accuracy_API)
    print(accuracy_opcode)
    print(accuracy_ensembel)
    print(accuracy_per_dynamic)
    print(accuracy_API_dynamic)
    print(accuracy_opcode_dynamic)
    print(accuracy_ensembel_dynamic)








