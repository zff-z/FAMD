import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 1.5})  # 设置图的宽度为12英寸，高度为4英寸，水平间距为0.5倍图宽

ax[0].plot(x, y1)
ax[1].plot(x, y2)
ax[2].plot(x, y3)

ax[0].set_title("Sine function")
ax[1].set_title("Cosine function")
ax[2].set_title("Sigmoid function")

fig.tight_layout()
plt.show()