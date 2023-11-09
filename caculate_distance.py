import numpy as np
import torch
from sklearn.cluster import KMeans
# from dynamic_prototypes import calculate_prototypes_Kmeans, calculate_dynamic_prototype

def calculate_prototypes_Kmeans(support_set, num_clusters):  #这个函数目前是正确的，不，是错误的，现在的问题是有可能会出现nan的值
    """
    计算每个类别的聚类原型
    Args:
        support_set: 支持集样本, shape=(num_support, feature_dim)
        labels: 支持集样本的标签, shape=(num_support,)
        num_clusters: 聚类的数量
    Returns:
        prototypes: 聚类原型, shape=(num_clusters, feature_dim)
    """
    feature_dim = len(support_set[0])
    prototypes = torch.zeros(num_clusters, feature_dim)
    support_set_np = np.array(support_set)
    kmeans = KMeans(n_clusters=num_clusters).fit(support_set_np)
    cluster_centers = kmeans.cluster_centers_
    for i in range(num_clusters):
        prototypes[i] = torch.from_numpy(cluster_centers[i])
    return prototypes

def calculate_dynamic_prototype(query, prototypes, num_clusters):
    distances = torch.zeros(num_clusters)

    for i in range(num_clusters):
        distance = np.linalg.norm(query - prototypes[i])
        distances[i] = distance
    weights = -distances  # 使用了softmax处理
    weights_softmax = torch.nn.functional.softmax(weights, dim=-1)
    dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(-1) * prototypes, dim=-2)

    return dynamic_prototypes

# def calculate_dynamic_prototype(query, prototypes, labels, num_clusters):
#     num_classes = prototypes.size(0)
#     num_queries = query.size(0)
#     distances = torch.zeros(num_queries, num_classes, num_clusters)
#     for c in range(num_classes):
#         class_distances = torch.cdist(query, prototypes[c])
#         distances[:, c] = class_distances
#     weights = -distances  #使用了softmax处理
#     weights_softmax = torch.nn.functional.softmax(weights, dim=-1)
#     dynamic_prototypes = torch.sum(weights_softmax.unsqueeze(-1) * prototypes.unsqueeze(0).unsqueeze(0),dim=-2)
#   #unsqueeze的用法是在指定维度增加1，dim=0表示在第一个维度上增加维度，dim等于1表示在第二个维度上增加1
#     return dynamic_prototypes

def calculate_prototype(family):
    vector_sum = np.sum(family, axis=0)
    average = vector_sum / len(family)

    return average

def calculate_euclidean_distance(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    return distance


def group_samples_by_label(support_set, labels):
    unique_labels = torch.unique(labels)
    num_families = len(unique_labels)
    families = [[] for _ in range(num_families)]

    for i in range(len(support_set)):
        label = labels[i]
        sample = support_set[i]
        family_index = (unique_labels == label).nonzero().item()
        families[family_index].append(sample)

    return families

def calculate_distances(families):
    num_families = len(families)
    avg_intra_distances = np.zeros(num_families) # 保存类内距离的平均值
    avg_inter_distances = np.zeros((num_families, num_families)) # 保存类间距离的平均值
    num_classes = torch.unique(labels).size(0)
    # 计算类内距离和类间距离
    for i in range(num_families):
        family_samples = families[i]
        # 计算类内距离
        intra_distances = []
        for sample in family_samples:
            # prototype = calculate_prototype(family_samples)
            k_means_cluster = calculate_prototypes_Kmeans(family_samples, 2)
            prototype = calculate_dynamic_prototype(sample, k_means_cluster, 2)
            distance = calculate_euclidean_distance(sample, prototype)
            intra_distances.append(distance)
        avg_intra_distances[i] = np.mean(intra_distances)

        # 计算类间距离
        for j in range(num_families):
            if j != i:
                other_family_samples = families[j]
                inter_distances = []
                for sample in family_samples:
                    prototype = calculate_prototype(other_family_samples)
                    distance = calculate_euclidean_distance(sample, prototype)
                    inter_distances.append(distance)
                avg_inter_distances[i][j] = np.mean(inter_distances)

    return avg_intra_distances, avg_inter_distances
