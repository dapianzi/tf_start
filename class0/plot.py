# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# cn enable
plt.rcParams['font.sans-serif'] = "Arial Unicode MS"
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# canvas
plt.figure(figsize=(10, 8), facecolor='lightgrey')
# add title
plt.suptitle('Matplotlib 示例')

n = 1024
dict_title = {
    "fontsize": 16,
    "color": "white",
    "backgroundcolor": "green"
}
dict_label = {
    "fontsize": 12,
    "color": "r"
}

# subplot 1
plt.subplot(221)
plt.title('1. 正态分布', fontdict=dict_title)
x1 = np.random.normal(0, 1, n)
y1 = np.random.normal(0, 1, n)
plt.scatter(x1, y1, color='b', marker='*')  # 绘制散点
plt.text(2.5, 2.5, "mean=0\nstd=1")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('横坐标x', fontdict=dict_label)
plt.ylabel('纵坐标y', fontdict=dict_label)

# subplot 2
plt.subplot(222)
plt.title('2. 均匀分布', fontdict=dict_title)
x2 = np.random.uniform(-4, 4, (1, n))
y2 = np.random.uniform(-4, 4, (1, n))
plt.scatter(x2, y2, color='r', marker='.')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('横坐标x', fontdict=dict_label)
plt.ylabel('纵坐标y', fontdict=dict_label)

# subplot 3
n = 24
plt.subplot(223)
plt.title('3. 温度与湿度', fontdict=dict_title)
y31 = np.random.randint(18, 30, n)
y32 = np.random.randint(40, 60, n)
# 绘制多个曲线
plt.plot(y31, label='温度')                    # 省略了x
plt.plot(y32, label='湿度')
plt.xlim(0, n-1)
plt.ylim(10, 70)
plt.xlabel('时间', fontdict=dict_label)
plt.ylabel('测量值', fontdict=dict_label)
plt.legend()                                    # 标示图例

# subplot 4
plt.subplot(224)
plt.title('4. 柱状图', fontdict=dict_title)
y41 = np.random.randint(15, 35, n)
y42 = np.random.randint(-35, -15, n)
plt.bar(range(n), y41, width=0.5, facecolor="lightgreen", edgecolor="green", label="收入")
plt.bar(range(n), y42, width=0.5, facecolor="orange", edgecolor="red", label="支出")
plt.legend()

plt.tight_layout()
plt.show()
