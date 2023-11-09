# import matplotlib.pyplot as plt
#
# weighted_static_prototype =   [0.8598, 0.9098, 0.9231, 0.9326, 0.9412, 0.9512, 0.9555, 0.9578, 0.9581, 0.9599] #无cluster
# weighted_dynamic_prototype_cluster2 = [0.9241, 0.9341, 0.9399, 0.9531, 0.960, 0.9640, 0.9632, 0.9679, 0.9699]   #cluster为2，2-10
# weighted_dynamic_prototype_cluster3   =       [0.9328, 0.9454, 0.9544, 0.961, 0.9654, 0.9664, 0.9723, 0.9732]   #cluster为3，3-10
# weighted_dynamic_prototype_cluster4 =                 [0.9378, 0.9480, 0.954, 0.9620, 0.9650, 0.9712, 0.9739]  #cluster为4,4-10
# # 创建一个新的图像
# fig, ax = plt.subplots()
#
#
# ax.plot(range(1, len(weighted_static_prototype)+1), weighted_static_prototype, label='Vanilla prototype (Without cluster)')
# ax.plot(range((11 - len(weighted_dynamic_prototype_cluster2)), 11), weighted_dynamic_prototype_cluster2, label='Cluster number t = 2')
# ax.plot(range((11 - len(weighted_dynamic_prototype_cluster3)), 11), weighted_dynamic_prototype_cluster3, label='Cluster number t = 3')
# ax.plot(range((11 - len(weighted_dynamic_prototype_cluster4)), 11), weighted_dynamic_prototype_cluster4, label='Cluster number t = 4')
# # 设置x轴和y轴标签
# ax.set_xlabel('Shot')
# ax.set_ylabel('Accuracy')
#
# # 设置图像标题
# ax.set_title('Model Accuracy on 5-way 5-shot')
# # 保存图像
# # fig.savefig('./cluster.pdf')
#
#
#
#
# # 添加图例，将标签与曲线对应起来，并放置在图的右上角
# ax.legend(loc='lower right')
#
# ax.grid(True)
# # 显示图像
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

weighted_static_prototype = [0.8598, 0.9098, 0.9231, 0.9326, 0.9412, 0.9512, 0.9555, 0.9578, 0.9581, 0.9599] #无cluster
weighted_dynamic_prototype_cluster2 = [0.9241, 0.9341, 0.9399, 0.9531, 0.960, 0.9640, 0.9632, 0.9679, 0.9699]   #cluster为2，2-10
weighted_dynamic_prototype_cluster3 = [0.9328, 0.9454, 0.9544, 0.961, 0.9654, 0.9664, 0.9723, 0.9732]   #cluster为3，3-10
weighted_dynamic_prototype_cluster4 = [0.9378, 0.9480, 0.954, 0.9620, 0.9650, 0.9712, 0.9739]  #cluster为4,4-10

# 创建一个新的图像
fig, ax = plt.subplots()

# 设置线条样式和颜色
line_styles = ['-', '--', '-.', ':']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# 绘制曲线
ax.plot(range(1, len(weighted_static_prototype) + 1), weighted_static_prototype, linestyle=line_styles[0], linewidth=1.5, marker='o', markersize=5, color=colors[0], label='Vanilla prototype (Without cluster)')
ax.plot(range((11 - len(weighted_dynamic_prototype_cluster2)), 11), weighted_dynamic_prototype_cluster2, linestyle=line_styles[1], linewidth=1.5, marker='s', markersize=5, color=colors[1], label='Cluster number t = 2')
ax.plot(range((11 - len(weighted_dynamic_prototype_cluster3)), 11), weighted_dynamic_prototype_cluster3, linestyle=line_styles[2], linewidth=1.5, marker='^', markersize=5, color=colors[2], label='Cluster number t = 3')
ax.plot(range((11 - len(weighted_dynamic_prototype_cluster4)), 11), weighted_dynamic_prototype_cluster4, linestyle=line_styles[3], linewidth=1.5, marker='D', markersize=5, color=colors[3], label='Cluster number t = 4')

# 设置x轴和y轴标签
ax.set_xlabel('Shot')
ax.set_ylabel('Accuracy')

# 设置图像标题
ax.set_title('Model Accuracy on 5-way 5-shot')



# 添加图例，将标签与曲线对应起来，并放置在图的右上角
ax.legend(loc='lower right')

# 设置背景网格样式
ax.grid(color='lightgray', linestyle='--')

# 自动调整子图的布局
plt.tight_layout()

# 显示图像
plt.show()

fig.savefig('./cluster.pdf')