import matplotlib.pyplot as plt

# 启用TeX字体样式只用于标题
plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='normal', style='normal')

# 创建示例数据
x = [1, 2, 3, 4]
y = [10, 5, 10, 5]

# 绘制图形
plt.plot(x, y)

# 设置图像标题，混合使用英文和LaTeX字体
plt.title('English Text with LaTeX Font: $T_{avg}+\eta H$', fontsize=16)

# 显示图形
plt.show()
