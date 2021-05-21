# coding=utf8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

boston_housing = tf.keras.datasets.boston_housing
(x, y), (_, _) = boston_housing.load_data(test_split=0)  # 测试集为空

print(np.shape(x), x[0])
titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT", "MEDV"]

plt.rcParams['font.sans-serif'] = "Arial Unicode MS"
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.figure(figsize=(12, 12))  # 画布尺寸 5x5
plt.suptitle("Bostom Housing Props & Price")
for i in range(13):
    plt.subplot(4, 4, (i + 1))
    plt.title(str(i + 1) + ". " + titles[i])
    plt.xlabel(titles[i])
    plt.ylabel("房价(千美元)")
    plt.scatter(x[:, i], y)

plt.tight_layout()
plt.show()
