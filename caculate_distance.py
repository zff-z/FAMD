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



support_set = torch.tensor([[-1.7787e-01, -1.7560e-01,  2.9593e-04,  2.3789e-01, -2.9365e-01,
         -2.0336e-01,  5.7687e-02, -2.1122e-01,  2.6243e-01,  6.2016e-01,
         -1.9778e-01, -1.4787e-01, -3.4602e-01, -3.8929e-01,  2.1936e-01,
         -4.0616e-01],
        [-1.0727e-01, -6.3720e-02,  6.1639e-02,  5.7036e-02, -8.3175e-02,
         -1.2988e-01,  1.0065e-01, -2.1526e-02, -1.5009e-01,  8.5681e-02,
          1.1661e-01, -1.7785e-01, -6.4790e-02, -8.8619e-02,  1.2960e-01,
          5.8504e-02],
        [-2.1504e-01, -1.2653e-01,  3.8708e-02,  1.8994e-01, -1.9177e-01,
         -1.8343e-01,  2.6770e-02, -1.2930e-01, -2.0731e-01,  1.2147e-01,
          1.0344e-01, -1.6594e-01, -3.8657e-02, -1.9521e-01,  2.2081e-01,
         -9.5603e-02],
        [-1.1471e-01,  9.2548e-02,  2.5087e-01, -2.6012e-01,  8.6566e-03,
         -3.1651e-01,  2.2432e-01, -5.4240e-02, -5.7240e-02,  5.3901e-02,
         -7.4519e-02, -4.0355e-01, -2.7653e-01, -3.0397e-02,  4.6418e-01,
         -5.9758e-02],
        [ 5.5096e-02, -8.3666e-02,  3.4167e-01, -1.5675e-01, -1.8574e-02,
          8.1462e-02, -1.1649e-01, -5.9063e-02,  3.6682e-02,  1.2311e-01,
          1.9002e-01, -3.3165e-01, -1.4990e-01, -1.5612e-01,  2.2665e-01,
         -1.3298e-01],
        [-1.7787e-01, -1.7560e-01,  2.9593e-04,  2.3789e-01, -2.9365e-01,
         -2.0336e-01,  5.7687e-02, -2.1122e-01,  2.6243e-01,  6.2016e-01,
         -1.9778e-01, -1.4787e-01, -3.4602e-01, -3.8929e-01,  2.1936e-01,
         -4.0616e-01],
        [ 2.8033e-03,  1.0487e-01,  2.0065e-01, -5.8419e-02, -1.2004e-01,
         -1.1567e-01,  1.5221e-01,  1.1804e-02,  4.7494e-02,  2.1955e-01,
         -1.9169e-01, -2.6570e-01, -2.5454e-01, -5.8479e-02,  2.2193e-01,
         -7.8045e-02],
        [-1.3835e-01, -8.4773e-02,  1.1682e-01,  2.4172e-02, -9.6172e-02,
         -1.1754e-01,  1.5721e-01,  2.1993e-02, -5.1437e-02,  1.8945e-01,
          3.5000e-02, -2.9935e-01, -1.8450e-01, -5.4413e-02,  2.2071e-01,
         -6.4404e-02],
        [-2.7591e-01, -5.9061e-02,  6.5775e-02,  1.3071e-01, -3.4925e-01,
         -2.4723e-01,  3.8219e-02, -1.7908e-01,  1.1453e-01,  4.7583e-01,
         -1.4007e-01, -2.1839e-01, -3.2405e-01, -2.9062e-01,  2.9186e-01,
         -3.6208e-01],
        [ 1.2385e-01,  1.8914e-01,  1.7218e-01, -1.1479e-01, -1.5903e-01,
         -1.5170e-01, -1.1519e-01, -2.5188e-01,  2.7496e-01,  2.5421e-01,
         -2.5219e-02,  6.3532e-02, -1.6519e-01, -3.2863e-01,  3.3813e-01,
         -3.0726e-01],
        [-3.7202e-02,  3.3279e-01,  1.4876e-01,  2.9211e-01, -5.0779e-01,
          1.4209e-01, -2.4443e-01, -7.1348e-02,  1.4328e-01,  1.5189e-01,
         -2.3800e-01,  1.0152e-01,  1.5581e-01, -1.5139e-01,  1.1052e-01,
         -4.8797e-01],
        [ 1.1615e-01,  2.0184e-01,  2.4497e-01, -1.2253e-01, -3.0813e-01,
          6.0704e-03, -1.7867e-01, -2.3969e-01,  2.8719e-01,  3.8447e-01,
         -7.6908e-02, -3.8785e-03, -2.1335e-01, -2.7150e-01,  2.1029e-01,
         -2.9072e-01],
        [ 5.5096e-02, -8.3666e-02,  3.4167e-01, -1.5675e-01, -1.8574e-02,
          8.1462e-02, -1.1649e-01, -5.9063e-02,  3.6682e-02,  1.2311e-01,
          1.9002e-01, -3.3165e-01, -1.4990e-01, -1.5612e-01,  2.2665e-01,
         -1.3298e-01],
        [-1.7787e-01, -1.7560e-01,  2.9593e-04,  2.3789e-01, -2.9365e-01,
         -2.0336e-01,  5.7687e-02, -2.1122e-01,  2.6243e-01,  6.2016e-01,
         -1.9778e-01, -1.4787e-01, -3.4602e-01, -3.8929e-01,  2.1936e-01,
         -4.0616e-01],
        [ 1.2385e-01,  1.8914e-01,  1.7218e-01, -1.1479e-01, -1.5903e-01,
         -1.5170e-01, -1.1519e-01, -2.5188e-01,  2.7496e-01,  2.5421e-01,
         -2.5219e-02,  6.3532e-02, -1.6519e-01, -3.2863e-01,  3.3813e-01,
         -3.0726e-01],
        [ 5.5096e-02, -8.3666e-02,  3.4167e-01, -1.5675e-01, -1.8574e-02,
          8.1462e-02, -1.1649e-01, -5.9063e-02,  3.6682e-02,  1.2311e-01,
          1.9002e-01, -3.3165e-01, -1.4990e-01, -1.5612e-01,  2.2665e-01,
         -1.3298e-01],
        [-2.8550e-01,  1.6067e-01,  1.9872e-01,  6.2837e-03, -2.6771e-01,
         -2.5368e-01,  2.2788e-02, -2.0354e-01, -2.9812e-01,  1.0436e-01,
         -4.2647e-02, -3.2922e-01, -1.4657e-01, -1.3142e-01,  3.6663e-01,
         -1.0977e-01],
        [-3.7202e-02,  3.3279e-01,  1.4876e-01,  2.9211e-01, -5.0779e-01,
          1.4209e-01, -2.4443e-01, -7.1348e-02,  1.4328e-01,  1.5189e-01,
         -2.3800e-01,  1.0152e-01,  1.5581e-01, -1.5139e-01,  1.1052e-01,
         -4.8797e-01],
        [-2.8912e-01,  2.4152e-01,  2.9477e-01,  1.5276e-01, -3.6386e-01,
         -8.4585e-02, -4.5151e-02, -1.6625e-01, -2.4796e-01,  1.8820e-01,
         -1.9597e-01, -3.2758e-01, -6.1165e-02, -1.6572e-01,  4.0756e-01,
         -3.1241e-01],
        [ 5.5096e-02, -8.3666e-02,  3.4167e-01, -1.5675e-01, -1.8574e-02,
          8.1462e-02, -1.1649e-01, -5.9063e-02,  3.6682e-02,  1.2311e-01,
          1.9002e-01, -3.3165e-01, -1.4990e-01, -1.5612e-01,  2.2665e-01,
         -1.3298e-01],
        [-1.7787e-01, -1.7560e-01,  2.9593e-04,  2.3789e-01, -2.9365e-01,
         -2.0336e-01,  5.7687e-02, -2.1122e-01,  2.6243e-01,  6.2016e-01,
         -1.9778e-01, -1.4787e-01, -3.4602e-01, -3.8929e-01,  2.1936e-01,
         -4.0616e-01],
        [-2.7793e-01,  2.1547e-01,  2.5747e-01,  4.8110e-02, -3.0138e-01,
         -1.5626e-01,  3.4132e-02, -1.3504e-01, -2.5036e-01,  1.5964e-01,
         -1.3389e-01, -3.6018e-01, -1.4367e-01, -1.0811e-01,  3.5513e-01,
         -1.8204e-01],
        [ 1.2385e-01,  1.8914e-01,  1.7218e-01, -1.1479e-01, -1.5903e-01,
         -1.5170e-01, -1.1519e-01, -2.5188e-01,  2.7496e-01,  2.5421e-01,
         -2.5219e-02,  6.3532e-02, -1.6519e-01, -3.2863e-01,  3.3813e-01,
         -3.0726e-01],
        [ 3.9555e-02, -7.0233e-02,  3.4624e-01, -1.6118e-01,  2.9249e-02,
          5.4090e-02, -1.0446e-01, -5.2366e-02, -2.0544e-02,  7.0108e-02,
          1.6903e-01, -3.5861e-01, -1.2059e-01, -1.1018e-01,  2.6003e-01,
         -1.2221e-01],
        [-2.1504e-01, -1.2653e-01,  3.8708e-02,  1.8994e-01, -1.9177e-01,
         -1.8343e-01,  2.6770e-02, -1.2930e-01, -2.0731e-01,  1.2147e-01,
          1.0344e-01, -1.6594e-01, -3.8657e-02, -1.9521e-01,  2.2081e-01,
         -9.5603e-02]])
labels = torch.tensor([3, 1, 1, 4, 0, 3, 4, 1, 3, 2, 1, 2, 0, 3, 2, 0, 4, 2, 4, 0, 3, 4, 2, 0,1])
families = group_samples_by_label(support_set, labels)

avg_intra_distances, avg_inter_distances = calculate_distances(families)
print(avg_intra_distances)
print(avg_inter_distances)