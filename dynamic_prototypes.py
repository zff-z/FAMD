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





support_set = torch.tensor([[-3.1077e-01,  4.5648e-01, -1.2750e-01, -1.5361e-01, -2.9748e-01,
         -1.6593e-01, -3.0531e-01, -7.6568e-02, -2.9097e-03,  2.1330e-01,
          4.6858e-01, -2.0245e-02,  9.9889e-02, -8.3943e-02,  2.1693e-01,
          4.8423e-01],[-3.1077e-01,  4.5648e-01, -1.2750e-01, -1.5361e-01, -2.9748e-01,
         -1.6593e-01, -3.0531e-01, -7.6568e-02, -2.9097e-03,  2.1330e-01,
          4.6858e-01, -2.0245e-02,  9.9889e-02, -8.3943e-02,  2.1693e-01,
          4.8423e-01],[-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02],
                            [-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02],[-2.4529e-01,  7.1456e-02, -7.8395e-02, -1.2615e-01, -7.0753e-02,
          4.5193e-02, -1.2240e-01, -2.4246e-02,  1.3116e-01,  2.5967e-02,
          1.7675e-01, -2.8080e-02, -8.7579e-02,  1.2324e-01,  3.6285e-02,
          6.0378e-02], [-2.7967e-01,  2.6587e-01,  1.7968e-01,  7.7580e-02, -1.1227e-01,
         -2.9175e-01, -7.4684e-02, -7.5092e-02, -6.4413e-02,  1.3653e-01,
          2.8298e-01,  1.5779e-01, -1.8527e-01,  1.4171e-01,  4.9685e-02,
          3.2615e-02], [ 3.2769e-02,  2.0347e-01,  1.5198e-01,  2.0442e-01, -7.7342e-02,
         -3.4048e-01, -8.8785e-03, -1.5651e-01, -1.9128e-01, -1.6242e-03,
          7.1127e-02,  1.0843e-01, -1.9196e-01,  1.0540e-02,  1.8742e-01,
          9.0601e-02],[-2.0374e-01,  4.2167e-01, -1.9597e-01, -2.7181e-01, -2.8789e-01,
         -9.1923e-02, -2.6904e-01, -5.5085e-02,  8.1183e-02,  1.2877e-01,
          4.3534e-01, -1.0716e-02,  1.6222e-01, -1.1568e-01,  2.1658e-01,
          4.7825e-01], [-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01], [-2.6563e-01,  1.8280e-01,  1.1985e-01,  9.5721e-02, -2.0591e-01,
         -2.8569e-01, -1.5566e-01, -1.5315e-01, -1.4067e-01,  2.2588e-01,
          2.1557e-01,  1.1845e-01, -2.1616e-01,  1.3904e-01,  8.0460e-02,
          3.8041e-02], [-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01], [-2.0374e-01,  4.2167e-01, -1.9597e-01, -2.7181e-01, -2.8789e-01,
         -9.1923e-02, -2.6904e-01, -5.5085e-02,  8.1183e-02,  1.2877e-01,
          4.3534e-01, -1.0716e-02,  1.6222e-01, -1.1568e-01,  2.1658e-01,
          4.7825e-01],[-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01], [-1.5247e-01,  1.6635e-01,  9.7378e-02,  1.8332e-01, -2.8069e-01,
         -3.3492e-01, -2.7097e-01, -1.4401e-01, -2.6361e-01,  3.5969e-01,
          1.7120e-01,  1.5134e-01, -1.6082e-01,  3.9155e-02,  5.1812e-02,
          2.7738e-01],
            [-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02], [-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01], [-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02],[ 1.3407e-02,  3.0827e-01,  1.5085e-01,  1.5897e-01, -2.1492e-01,
         -3.7781e-01, -1.1726e-01, -1.4028e-01, -1.4798e-01,  4.2030e-04,
          7.7679e-02,  8.9229e-02, -1.8097e-01, -4.3187e-02,  2.1395e-01,
          2.8229e-01],[-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01],[-2.6563e-01,  1.8280e-01,  1.1985e-01,  9.5721e-02, -2.0591e-01,
         -2.8569e-01, -1.5566e-01, -1.5315e-01, -1.4067e-01,  2.2588e-01,
          2.1557e-01,  1.1845e-01, -2.1616e-01,  1.3904e-01,  8.0460e-02,
          3.8041e-02],[-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02],[-3.1077e-01,  4.5648e-01, -1.2750e-01, -1.5361e-01, -2.9748e-01,
         -1.6593e-01, -3.0531e-01, -7.6568e-02, -2.9097e-03,  2.1330e-01,
          4.6858e-01, -2.0245e-02,  9.9889e-02, -8.3943e-02,  2.1693e-01,
          4.8423e-01],[-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01],[-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01],[-1.5247e-01,  1.6635e-01,  9.7378e-02,  1.8332e-01, -2.8069e-01,
         -3.3492e-01, -2.7097e-01, -1.4401e-01, -2.6361e-01,  3.5969e-01,
          1.7120e-01,  1.5134e-01, -1.6082e-01,  3.9155e-02,  5.1812e-02,
          2.7738e-01]])
labels = torch.tensor([0, 0, 3, 3, 2, 4, 4, 0, 1, 4, 2, 0, 2, 1, 3, 1, 3, 4, 1, 4, 3, 0, 2, 2,1])
query = torch.tensor([[-2.7967e-01,  2.6587e-01,  1.7968e-01,  7.7580e-02, -1.1227e-01,
         -2.9175e-01, -7.4684e-02, -7.5092e-02, -6.4413e-02,  1.3653e-01,
          2.8298e-01,  1.5779e-01, -1.8527e-01,  1.4171e-01,  4.9685e-02,
          3.2615e-02],[ 4.8981e-02,  3.4336e-01,  3.1060e-01,  2.4693e-01, -1.9541e-01,
         -4.5070e-01, -3.5525e-02, -1.4181e-01, -2.0120e-01,  1.2911e-01,
          1.6692e-01,  3.0324e-01, -1.8774e-01,  6.3754e-02,  1.0631e-01,
          1.7537e-01],[-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01],[-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01],[-2.3027e-01,  1.7431e-01,  1.7482e-02, -6.7367e-02, -1.8047e-01,
         -1.1954e-01, -1.1995e-01, -1.4884e-01,  2.5385e-02,  7.4930e-02,
          1.5188e-01, -7.8538e-03, -1.3064e-01,  1.5267e-01,  8.6284e-02,
          5.4770e-02],[-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01],[-9.6525e-02,  2.5342e-01,  1.6048e-01,  1.8349e-01, -1.4847e-02,
         -3.3957e-01,  3.2273e-02,  1.4510e-01, -1.6616e-01,  1.4959e-01,
          3.5412e-01,  2.0744e-01, -2.1523e-01, -7.7235e-02,  1.6747e-01,
          2.7579e-01],[-2.0662e-01,  2.7494e-01,  1.5381e-01,  1.7115e-01, -1.5928e-01,
         -3.5873e-01, -8.0877e-02, -2.3337e-01, -1.5740e-01,  1.4693e-01,
          2.2962e-01,  1.4512e-01, -1.6829e-01,  1.0741e-01,  1.2372e-01,
          3.3745e-02],[-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01],[-3.1077e-01,  4.5648e-01, -1.2750e-01, -1.5361e-01, -2.9748e-01,
         -1.6593e-01, -3.0531e-01, -7.6568e-02, -2.9097e-03,  2.1330e-01,
          4.6858e-01, -2.0245e-02,  9.9889e-02, -8.3943e-02,  2.1693e-01,
          4.8423e-01],[-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01],[-1.4150e-01,  1.5076e-01,  1.3324e-01,  1.6008e-01, -2.4982e-02,
         -2.5606e-01,  2.0470e-02, -1.2706e-01, -6.4230e-02, -8.1346e-02,
          4.6543e-02,  1.2241e-03, -2.2801e-01,  1.2082e-01,  1.9237e-01,
         -9.4528e-02],[-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02],[-1.2755e-01,  1.1525e-01, -5.8211e-02,  9.2298e-03, -1.5929e-02,
         -8.3207e-02, -1.0681e-01,  7.7513e-02, -4.5120e-02,  1.2464e-01,
          2.1891e-01,  6.2529e-02, -6.6117e-02, -6.0695e-02,  7.5529e-02,
          1.7843e-01],[-2.7967e-01,  2.6587e-01,  1.7968e-01,  7.7580e-02, -1.1227e-01,
         -2.9175e-01, -7.4684e-02, -7.5092e-02, -6.4413e-02,  1.3653e-01,
          2.8298e-01,  1.5779e-01, -1.8527e-01,  1.4171e-01,  4.9685e-02,
          3.2615e-02],[-2.2195e-01,  1.1299e-01,  3.6121e-02,  1.2948e-01, -4.5395e-02,
         -2.3072e-01, -1.1391e-01,  2.2982e-01, -1.8948e-01,  3.2646e-01,
          3.4634e-01,  1.8596e-01, -1.8595e-01, -9.0234e-02,  3.9933e-02,
          3.1389e-01],
            [-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01], [-2.7967e-01,  2.6587e-01,  1.7968e-01,  7.7580e-02, -1.1227e-01,
         -2.9175e-01, -7.4684e-02, -7.5092e-02, -6.4413e-02,  1.3653e-01,
          2.8298e-01,  1.5779e-01, -1.8527e-01,  1.4171e-01,  4.9685e-02,
          3.2615e-02],[-2.7811e-02,  2.1146e-01,  2.0946e-01,  4.1484e-01, -2.3691e-01,
         -4.5149e-01, -1.9386e-01, -1.3233e-01, -3.8293e-01,  2.7891e-01,
          7.2911e-02,  2.4007e-01, -2.1796e-01, -3.8084e-02,  1.4237e-01,
          2.8992e-01],[-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02], [-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01],[-3.1077e-01,  4.5648e-01, -1.2750e-01, -1.5361e-01, -2.9748e-01,
         -1.6593e-01, -3.0531e-01, -7.6568e-02, -2.9097e-03,  2.1330e-01,
          4.6858e-01, -2.0245e-02,  9.9889e-02, -8.3943e-02,  2.1693e-01,
          4.8423e-01], [-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02],[-6.9899e-02,  3.1929e-01,  7.0615e-02,  1.5474e-01,  1.7844e-01,
         -2.8806e-01,  1.9120e-01, -9.1535e-02, -4.8606e-03, -1.8236e-01,
          1.9704e-01, -2.6096e-02, -5.7412e-02, -3.1804e-02,  2.7540e-01,
         -8.6366e-02], [-7.2510e-02,  5.5831e-03, -5.6222e-02, -2.9476e-04, -8.6363e-02,
          2.0746e-02, -1.4381e-01, -5.8073e-02, -1.1580e-02,  4.7181e-02,
          5.4780e-02, -1.1218e-02, -1.5117e-01,  7.8301e-02,  5.4542e-02,
          1.4430e-01]])
output_size = 16
prototypes_Kmeans = calculate_prototypes_Kmeans(support_set, labels, 2)
# prototypes_DBSCAN = calculate_prototypes_DBSCAN(support_set, labels)
print(prototypes_Kmeans.shape)
# print(prototypes_DBSCAN.shape)
dynamic_prototypes = calculate_dynamic_prototype(query, prototypes_Kmeans, labels, 2)
print(dynamic_prototypes.shape)