"""
使用所有特征做 多元线性回归 预测房价
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load data
boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()
print(train_x.shape, train_y.shape)

train_n = len(train_x)
test_n = len(test_x)

# processing data
# normalization
train_x = (train_x - train_x.min(axis=0)) / (train_x.max(axis=0) - train_x.min(axis=0))
test_x = (test_x - test_x.min(axis=0)) / (test_x.max(axis=0) - test_x.min(axis=0))

# (1, x1, x2, x3, ..., xn) * (w0, w1, w2, w3, ..., wn) = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + w0
# w1,w2,w3,...,w4   => 特征系数 => 一元回归中的w
# w0                => 偏移量   => 一元回归中的b
train_set = tf.cast(tf.concat([np.ones(train_n).reshape(-1, 1), train_x], axis=1), tf.float32)  # 开头拼接一列全1向量
test_set = tf.cast(tf.concat([np.ones(test_n).reshape(-1, 1), test_x], axis=1), tf.float32)
train_label = tf.constant(train_y.reshape(-1, 1), tf.float32)  # 转为列向量
test_label = tf.constant(test_y.reshape(-1, 1), tf.float32)

print(train_set.shape, test_set.shape)

# super variables
learn_rate = 0.01
iter_times = 3000
display_step = 150

w = tf.Variable(np.random.randn(14, 1), dtype=tf.float32)  # 初始化参数
b = tf.Variable(np.random.randn(1, ), dtype=tf.float32)

print(w.shape, b.shape)

# trainning
mse_train = []
mse_test = []

for i in range(0, iter_times + 1):
    with tf.GradientTape() as tape:
        # 定义函数
        pred_train = tf.matmul(train_set, w) + b
        loss_train = 0.5 * tf.reduce_mean(tf.square(train_label - pred_train))

        pred_test = tf.matmul(test_set, w) + b
        loss_test = 0.5 * tf.reduce_mean(tf.square(test_label - pred_test))

    # 记录均方误差
    mse_train.append(loss_train)
    mse_test.append(loss_test)

    # 以loss_train的表达式对[w,b]求偏导数，算出梯度
    dL_dw, dL_db = tape.gradient(loss_train, [w, b])
    # 后向传播，根据学习率更新参数
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % display_step == 0:
        # print progress
        print("i: %i, Train loss: %f, Test Loss: %f" % (i, loss_train, loss_test))

plt.figure(figsize=(12, 8))
# plot1. data vs model


# plot2. loss
plt.subplot(222)
plt.plot(mse_train, color="blue", linewidth=3, label="train_loss")
plt.plot(mse_test, color="red", linewidth=1.5, label="test loss")
plt.legend(loc="upper right")

# plot3. predict with train dataset
plt.subplot(223)
plt.plot(train_label, color="blue", marker="o", label="true price")
plt.plot(pred_train, color="red", marker="x", label="predict price")
plt.legend()

# plot4. predict with test dataset
plt.subplot(224)
plt.plot(test_label, color="blue", marker="o", label="true price")
plt.plot(pred_test, color="red", marker="x", label="predict price")
plt.legend()

plt.show()
