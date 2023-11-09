import matplotlib.pyplot as plt

# # 每个epoch的准确度
# accuracy_per = [0.6372, 0.6532, 0.6596, 0.6668, 0.6544, 0.6676, 0.642, 0.6516, 0.6832, 0.6944]
# accuracy_API = [0.6256, 0.6412, 0.6688, 0.6392, 0.6504, 0.6576, 0.6532, 0.6596, 0.668, 0.666]
# accuracy_opcode = [0.5324, 0.5924, 0.5744, 0.5688, 0.5672, 0.5876, 0.5836, 0.5652, 0.592, 0.5684]
# accuracy_p_A_o = [0.65, 0.67, 0.683, 0.682, 0.671, 0.681, 0.694, 0.666, 0.685, 0.704]
# accuracy_ensembel_without_weighted = [0.8309, 0.8257, 0.8265, 0.8298, 0.8264, 0.8397, 0.8401, 0.8296, 0.8361, 0.8264]
# accuracy_ensembel_with_weighted = [0.8528, 0.835, 0.8484, 0.832, 0.8428, 0.8447, 0.8523, 0.8358, 0.8465, 0.8482]
# accuracy_ensembel_with_weighted_5way_7support = [0.8662, 0.8533, 0.874, 0.852, 0.844, 0.8653, 0.8463, 0.8637, 0.872, 0.8593]
weighted_static_prototype = [0.8598, 0.8436, 0.8371, 0.8404, 0.8404, 0.838, 0.8504, 0.8437, 0.854, 0.8444]
weighted_dynamic_prototype_cluster2 = [0.8598, 0.8596, 0.8567, 0.8576, 0.852, 0.8544, 0.8632, 0.8522, 0.8612, 0.8636]   #cluster为2，会高1%-2%
weighted_dynamic_prototype_cluster3   = [0.8598, 0.8864, 0.9328, 0.9454, 0.9544, 0.961, 0.9654, 0.9702, 0.9760, 0.9765]   #cluster为3，会高2%-3%左右
weighted_dynamic_prototype_cluster4 = [0.8598, 0.8668, 0.8628, 0.8626, 0.8568, 0.8416, 0.8506, 0.8499, 0.864, 0.8494]  #cluster为4
# 创建一个新的图像
fig, ax = plt.subplots()

# 绘制三条曲线，分别使用不同的颜色和标签
# ax.plot(range(1, len(accuracy_per)+1), accuracy_per, label='Accuracy_per')
# ax.plot(range(1, len(accuracy_API)+1), accuracy_API, label='Accuracy_API')
# ax.plot(range(1, len(accuracy_opcode)+1), accuracy_opcode, label='Accuracy_opcode')
ax.plot(range(1, len(weighted_static_prototype)+1), weighted_static_prototype, label='t = 1 (Without cluster)')
ax.plot(range(1, len(weighted_dynamic_prototype_cluster2)+1), weighted_dynamic_prototype_cluster2, label='t = 2')
ax.plot(range(1, len(weighted_dynamic_prototype_cluster3)+1), weighted_dynamic_prototype_cluster3, label='t = 3')
ax.plot(range(1, len(weighted_dynamic_prototype_cluster4)+1), weighted_dynamic_prototype_cluster4, label='t = 4')
# accuracy_opcode = [0.752, 0.696, 0.680, 0.560, 0.552, 0.460, 0.440, 0.420, 0.418, 0.408]
# accuracy_API = [0.760, 0.728, 0.728, 0.616, 0.504, 0.456, 0.448, 0.392, 0.368, 0.332]
# accuracy_per = [0.760, 0.744, 0.704, 0.664, 0.552, 0.480, 0.440, 0.439, 0.432, 0.368]
# fig, ax = plt.subplots()
# ax.plot(range(1, len(accuracy_per)+1), accuracy_per, label='Accuracy_per')
# ax.plot(range(1, len(accuracy_API)+1), accuracy_API, label='Accuracy_API')
# ax.plot(range(1, len(accuracy_opcode)+1), accuracy_opcode, label='Accuracy_opcode')
# 设置x轴和y轴标签
ax.set_xlabel('Shot')
ax.set_ylabel('Accuracy')

# 设置图像标题
ax.set_title('Model Accuracy on 5-way 5-shot')





# 添加图例，将标签与曲线对应起来，并放置在图的右上角
ax.legend(loc='upper right')
# ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
# ax.legend(bbox_to_anchor=(1.0, 1.0))
# fig.set_size_inches(8, 6)
ax.grid(True)
# 显示图像
plt.tight_layout()
plt.show()
