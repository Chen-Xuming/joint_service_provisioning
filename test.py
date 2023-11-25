import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果的可重复性
np.random.seed(42)

# 生成数据
x_values = np.arange(40, 101, 10)
labels = ['A', 'B', 'C', 'D']
data = {label: np.random.uniform(5, 20, len(x_values)) for label in labels}
errors = {label: np.random.uniform(2, 4, len(x_values)) for label in labels}

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 逐根绘制柱子和误差棒
width = 0.2  # 柱的宽度
for i, x in enumerate(x_values):
    for label in labels:
        # 计算x偏移量
        x_offset = i * len(labels) * width
        # 绘制柱子和误差棒
        ax.bar(x + x_offset, data[label][i], width=width, label=label, yerr=errors[label][i], capsize=4)

# 设置图形属性
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_title('逐根绘制柱状图和误差棒')
ax.set_xticks(x_values + width * ((len(labels) * len(x_values)) - 1) / 2)
ax.set_xticklabels(x_values)
ax.legend()

# 显示图形
plt.show()
