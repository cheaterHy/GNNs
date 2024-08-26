from matplotlib import pyplot as plt
import pickle as pkl
import torch
import numpy as np
from utils import *

num_fea = 3
data = load_data(args.dataset,num_fea)
feature = data.x
train_mask = data.train_mask
print(feature.shape, train_mask.shape)

x = [point[0] for point in feature]
y = [point[1] for point in feature]
z = [point[2] for point in feature]
colors = train_mask

# plt.scatter(x, y, c=colors, cmap='coolwarm', alpha=0.5)  # 根据 colors 绘制散点图，并使用 'coolwarm' 颜色映射
# plt.colorbar()  # 添加颜色条
#
# plt.xlabel('feature 0')  # 添加 x 轴标签
# plt.ylabel('feature 1')  # 添加 y 轴标签
#
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 添加 3D 图形的坐标轴

# 根据 colors 绘制散点图，并使用 'coolwarm' 颜色映射
scatter = ax.scatter(x, y, z, c=colors, cmap='coolwarm', alpha=0.5)

plt.colorbar(scatter)  # 添加颜色条

ax.set_xlabel('feature 0')  # 添加 x 轴标签
ax.set_ylabel('feature 1')  # 添加 y 轴标签
ax.set_zlabel('feature 2')  # 添加 z 轴标签

plt.show()