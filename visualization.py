import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset import Per_API_Opcode_Dataset, CustomDataset
from ensemble_learning import ProtoNet
import torch
import seaborn as sns

# 加载数据集
dataset = Per_API_Opcode_Dataset('/home/zf/siamese_chatGPT/dataset/Test/CIC2019_per_API_opcode_5families.csv')
device = torch.device('cpu')
# 加载编码器
model_per = ProtoNet(57, 114, 16).to(device)
model_per.load_state_dict(torch.load('./model/model_right/Siamese_per.pth'))
model_per.to(device)

model_API = ProtoNet(78, 156, 16).to(device)
model_API.load_state_dict(torch.load('./model/model_right/Siamese_API.pth'))
model_API.to(device)

model_opcode = ProtoNet(343, 200, 16).to(device)
model_opcode.load_state_dict(torch.load('./model/model_right/Siamese_opcode.pth'))
model_opcode.to(device)


# 分别获取三种特征和标签
per_features, API_features, opcode_features = [], [], []
labels = []
for i in range(len(dataset)):
    (per, API, opcode), label = dataset[i]
    per_features.append(model_per(torch.from_numpy(per)))
    API_features.append(model_API(torch.from_numpy(API)))
    opcode_features.append(model_opcode(torch.from_numpy(opcode)))
    labels.append(label)



# 将每个Tensor转换为Numpy数组
per_features_np = np.array([f.detach().numpy() for f in per_features])
API_features_np = np.array([f.detach().numpy() for f in API_features])
opcode_features_np = np.array([f.detach().numpy() for f in opcode_features])


# 使用t-SNE将每种特征映射到二维平面上
tsne = TSNE(n_components=2, perplexity=8, learning_rate=200, n_iter=1000, random_state=5) # 初始化t-SNE模型
per_tsne = tsne.fit_transform(per_features_np)
API_tsne = tsne.fit_transform(API_features_np)
opcode_tsne = tsne.fit_transform(opcode_features_np)

# # 可视化每种特征
# plt.figure(figsize=(16, 5))
#
# # 标记形状设置
# markers = ['o', 's', '^', 'D', 'v', 'p']
#
# # 第一个子图：Siamese loss-based
# plt.subplot(1, 3, 1)
# for i, label in enumerate(set(labels)):
#     idx = np.where(labels == label)[0]
#     plt.scatter(per_tsne[idx, 0], per_tsne[idx, 1], marker=markers[i % len(markers)], s=15, alpha=0.6)
#
# plt.title('Siamese loss-based')
# plt.legend()
#
# # 第二个子图：Triplet loss-based
# plt.subplot(1, 3, 2)
# for i, label in enumerate(set(labels)):
#     idx = np.where(labels == label)[0]
#     plt.scatter(API_tsne[idx, 0], API_tsne[idx, 1], marker=markers[i % len(markers)], s=15, alpha=0.6)
#
# plt.title('Triplet loss-based')
# plt.legend()
#
# # 第三个子图：Softmax loss-based
# plt.subplot(1, 3, 3)
# for i, label in enumerate(set(labels)):
#     idx = np.where(labels == label)[0]
#     plt.scatter(opcode_tsne[idx, 0], opcode_tsne[idx, 1], marker=markers[i % len(markers)], s=15, alpha=0.6)
#
# plt.title('Softmax loss-based')
# plt.legend()



fig, ax = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 0.5})

# 标记形状设置
markers = ['o', 's', '^', 'D', 'v', 'p']

# 第一个子图：Siamese loss-based
# plt.subplot(1, 3, 1)
for i, label in enumerate(set(labels)):
    idx = np.where(labels == label)[0]
    ax[0].scatter(per_tsne[idx, 0], per_tsne[idx, 1], marker=markers[i % len(markers)], s=15, alpha=0.7,edgecolor='none')
    # ax[0].axis('off')
# plt.title('Siamese loss-based')
# plt.legend()

# 第二个子图：Triplet loss-based
# plt.subplot(1, 3, 2)
for i, label in enumerate(set(labels)):
    idx = np.where(labels == label)[0]
    ax[1].scatter(API_tsne[idx, 0], API_tsne[idx, 1], marker=markers[i % len(markers)], s=15, alpha=0.7,edgecolor='none')
    # ax[1].axis('off')
# plt.title('Triplet loss-based')
# plt.legend()

# 第三个子图：Softmax loss-based
# plt.subplot(1, 3, 3)
for i, label in enumerate(set(labels)):
    idx = np.where(labels == label)[0]
    ax[2].scatter(opcode_tsne[idx, 0], opcode_tsne[idx, 1], marker=markers[i % len(markers)], s=15, alpha=0.7,edgecolor='none')
    # ax[2].axis('off')
# plt.title('Softmax loss-based')
# plt.legend()

ax[0].set_title("Siamese loss-based")
ax[1].set_title("Triplet loss-based")
ax[2].set_title("Softmax loss-based")

fig.tight_layout()
# plt.subplots_adjust(wspace=4.0)
# 去除边框和导航栏
for ax in plt.gcf().get_axes():
    ax.axis('off')

# 保存图像
plt.savefig('./myplot_without_daohang.pdf')

# 显示图像
plt.show()


