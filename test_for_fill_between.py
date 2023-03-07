import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x1 = [1, 3, 5]
y1 = [7, 8, 9]
x2 = [2, 4, 6]
y2 = [10, 11, 12]

# 对数据进行排序
x = sorted(set(x1 + x2))
y1 = [y1[x1.index(xi)] if xi in x1 else None for xi in x]
y2 = [y2[x2.index(xi)] if xi in x2 else None for xi in x]

# 绘制两条曲线
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# 使用fill_between填充颜色
plt.fill_between(x, y1, y2=y2, alpha=0.5)

# 设置图例和标签
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fill between two lines with aligned x-axis')

# 显示图形
plt.savefig("analysis/test.pdf")