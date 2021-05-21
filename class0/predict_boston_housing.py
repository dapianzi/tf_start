"""
使用房间面积做一元线性回归预测房价
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load data
boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()
print(train_x.shape, train_y.shape)

# processing data
train_set = train_x[:, 5]  # chose RM
train_label = train_y
test_set = test_x[:, 5]
test_label = test_y

# super variables
learn_rate = 0.002
iter_times = 1000
display_step = 50

w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# trainning
mse_train = []
mse_test = []

for i in range(0, iter_times + 1):
    with tf.GradientTape() as tape:
        # 定义函数
        pred_train = w * train_set + b
        loss_train = 0.5 * tf.reduce_mean(tf.square(train_label - pred_train))

        pred_test = w * test_set + b
        loss_test = 0.5 * tf.reduce_mean(tf.square(test_label - pred_test))

    # 保存均方误差
    mse_train.append(loss_train)
    mse_test.append(loss_test)

    # 以loss_train的表达式对[w,b]求偏导数，算出梯度
    dL_dw, dL_db = tape.gradient(loss_train, [w, b])
    # 根据学习率更新参数
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % display_step == 0:
        # print progress
        print("i: %i, Train loss: %f, Test Loss: %f" % (i, loss_train, loss_test))

plt.figure(figsize=(15, 10))
# plot1. data vs model
plt.subplot(221)
plt.scatter(train_set, train_label, color="blue", label="data")
plt.plot(train_set, pred_train, color="red", label="model")
plt.legend(loc="upper left")

# plot2. loss
plt.subplot(222)
plt.plot(mse_train, color="blue", linewidth=3, label="train_loss")
plt.plot(mse_test, color="red", linewidth=3, label="test loss")
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
