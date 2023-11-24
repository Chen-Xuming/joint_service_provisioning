import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果的可重复性
np.random.seed(42)

# 生成数据
x_values = np.arange(40, 101, 10)
labels = ['A', 'B', 'C', 'D', 'E', 'F']
data = {label: np.random.uniform(5, 20, len(x_values)) for label in labels}
errors = {label: np.random.uniform(2, 4, len(x_values)) for label in labels}

print(data)
print(errors)

# 绘制柱状图和误差棒
fig, ax = plt.subplots()
width = 1.2  # 将柱子的宽度调大3倍

for i, label in enumerate(labels):
    x_offset = i * width
    ax.bar(x_values + x_offset, data[label], width=width, label=label, yerr=errors[label], capsize=4)

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('.....')
ax.set_xticks(x_values + width * (len(labels) - 1) / 2)
ax.set_xticklabels(x_values)
ax.legend()

# 显示图形
plt.show()
