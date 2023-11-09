import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from dataset import Per_API_Opcode_Dataset, CustomDataset
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def compute_prototypes(support_set, labels):  # 计算方式:直接取类中样本的平均值
    unique_labels = torch.unique(labels)
    prototypes = torch.zeros(len(unique_labels), output_size).to(support_set.device)

    for i, label in enumerate(unique_labels):
        prototypes[i] = support_set[labels == label].mean(dim=0)

    return prototypes

def calculate_prototypes_Kmeans(support_set, labels, num_clusters):  #这个函数目前是正确的，不，是错误的，现在的问题是有可能会出现nan的值
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
    for c in range(num_classes):  #每一个类
        class_prototypes = torch.zeros(num_clusters, support_set.size(1))
        class_support_set = support_set[labels == c]
        kmeans = KMeans(n_clusters=num_clusters).fit(class_support_set)
        cluster_centers = kmeans.cluster_centers_
        for i in range(num_clusters):
            # cluster_samples = class_support_set[kmeans.labels_ == i]  #这是选择所有簇里面的样本取平均作为样本中心
            # prototype = cluster_samples.mean(dim=0)
            # class_prototypes[i] = prototype
            class_prototypes[i] = torch.from_numpy(cluster_centers[i])  #直接选择样本中心
        # prototypes[c] = torch.from_numpy(class_prototypes)
        prototypes[c] = class_prototypes
    return prototypes

def calculate_prototypes_DBSCAN(support_set, labels):
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


def calculate_dynamic_prototype(query, prototypes, labels, num_clusters):
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
        distances[:, c] = class_distances
    #如果要动态计算每个类的原型，那么需要对于每个query的每个class都计算一下，for query, for class
    # weights = 1 / distances   #这里使用的是归一化处理
    # weights_sum = weights.sum(dim=-1, keepdim=True)
    # weights_sum[weights_sum == 0] = 1e-8
    # weights_normalized = weights / weights_sum
    # dynamic_prototypes = torch.sum(weights_normalized.unsqueeze(1) * prototypes.unsqueeze(0), dim=2)
    weights = -distances  #使用了softmax处理
    weights_softmax = torch.nn.functional.softmax(weights, dim=-1)
    # print(weights_softmax.shape) #torch.Size([25, 5, 2])  num_query * num_class * cluster
    # print(prototypes.shape)  #torch.Size([5, 2, 16])    num_class * cluster * feature_dim
    # dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(1) * prototypes.unsqueeze(0), dim=2)
    dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(-1) * prototypes.unsqueeze(0).unsqueeze(0),dim=-2)
  #unsqueeze的用法是在指定维度增加1，dim=0表示在第一个维度上增加维度，dim等于1表示在第二个维度上增加1
    return dynamic_prototypes  #shape=(num_query, num_classes, feature_dim)
