"""
二元逻辑回归
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import utils

# 预置参数
plt.rcParams['font.sans-serif'] = "Arial Unicode MS"
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.figure(figsize=(8, 12))

cm_pt = mpl.colors.ListedColormap(['royalblue', 'orangered'])
cm_bg = mpl.colors.ListedColormap(['cyan', 'gold'])

iris_train = utils.get_arr_from_csv('iris_training.csv')
iris_test = utils.get_arr_from_csv('iris_test.csv')


# 处理数据集
def pre_precess(dataset):
    x = dataset[:, :2]  # 提取前2列特征
    y = dataset[:, 4]  # 提取标签
    # 过滤 label=2 的数据
    x = x[y < 2]
    y = y[y < 2]
    # 均值化
    x = x - np.mean(x, axis=0)
    return x, y


def plot_class_line(x, w):
    xs = [np.min(x), np.max(x)]
    ys = -(w[1] * xs + w[0]) / w[2]
    plt.plot(xs, ys)


def plot_class_contourf(x, w):
    num = 300
    _max = np.max(x, axis=0)
    _min = np.min(x, axis=0)
    plt.axis([_min[0], _max[0], _min[1], _max[1]])
    m1, m2 = np.meshgrid(np.linspace(_min[0], _max[0], num), np.linspace(_min[1], _max[1], num))

    # Wrong with contourf!!
    # plt.contourf(m1, m2, (m1 * w[1] + m2 * w[2] + w[0]), num, cmap=cm_bg)

    mesh_x = tf.cast(np.stack((np.ones(num * num), m1.reshape(-1), m2.reshape(-1)), axis=1), dtype=tf.float32)
    mesh_y = tf.cast(1 / (1 + tf.exp(-tf.matmul(mesh_x, W))), tf.float32)
    mesh_y = tf.where(mesh_y < 0.5, 0, 1)
    mesh_c = tf.reshape(mesh_y, m1.shape)
    plt.pcolormesh(m1, m2, mesh_c, cmap=cm_bg, shading='auto')


train_x, train_y = pre_precess(iris_train)
test_x, test_y = pre_precess(iris_test)

axis_max = np.max(train_x, axis=0)
axis_min = np.min(train_x, axis=0)
n = len(train_x)

x0 = np.ones(n).reshape(-1, 1)
# (1, x1, x2) * (w0, w1, w2) = w1*x1 + w2*x2 + w0
# w1,w2 => 特征系数 => 一元回归中的w
# w0    => 偏移量   => 一元回归中的b
X = tf.cast(tf.concat((x0, train_x), axis=1), tf.float32)
Y = tf.cast(train_y.reshape(-1, 1), tf.float32)
print(X.shape, Y.shape)

X_test = tf.cast(tf.concat((np.ones(len(test_x)).reshape(-1, 1), test_x), axis=1), tf.float32)
Y_test = tf.cast(test_y.reshape(-1, 1), tf.float32)

plt.subplot(321)
plt.title('训练集特征分布')
plt.axis([axis_min[0], axis_max[0], axis_min[1], axis_max[1]])
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cm_pt)

# 超参数
lr = 0.2
epoch = 200
step = 20

W = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)
plot_class_line(X, W)

cross_entropy = []
accuracy = []
test_ce = []
test_acc = []

for i in range(epoch + 1):
    with tf.GradientTape() as tape:
        pred = 1 / (1 + tf.exp(-tf.matmul(X, W)))
        # 负号：熵与交叉熵的定义
        loss = -tf.reduce_mean(Y * tf.math.log(pred) + (1 - Y) * tf.math.log(1 - pred))
    pred_test = 1 / (1 + tf.exp(-tf.matmul(X_test, W)))
    loss_test = -tf.reduce_mean(Y_test * tf.math.log(pred_test) + (1 - Y_test) * tf.math.log(1 - pred_test))

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred > 0.5, 1., 0.), Y), tf.float32))
    acc_test = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_test > 0.5, 1., 0.), Y_test), tf.float32))

    cross_entropy.append(loss)
    accuracy.append(acc)
    test_ce.append(loss_test)
    test_acc.append(acc_test)

    dW = tape.gradient(loss, W)
    W.assign_sub(lr * dW)

    if i % step == 0:
        print(f"epoch {i}, Acc: {acc}, Loss: {loss}")
        plot_class_line(X, W)

plt.subplot(322)
plt.title('测试集特征分布')
plt.axis([axis_min[0], axis_max[0], axis_min[1], axis_max[1]])
plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap=cm_pt)

# 训练过程
plt.subplot(323)
plt.title('Train & Test Loss')
plt.plot(cross_entropy, color='cyan', label='Train Loss')
plt.plot(test_ce, color='red', label='Test Loss')
plt.legend()

plt.subplot(324)
plt.title('Train & Test Acc')
plt.plot(accuracy, color='cyan', label='Train Acc')
plt.plot(test_acc, color='red', label='Test Acc')
plt.legend()

n = 20
# 分类结果填充图-- 训练集
plt.subplot(325)
plt.title('Contourf-Train')
# 绘制分类背景填充
# 绘制分类背景填充
plot_class_contourf(train_x, W)
# 绘制数据点
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cm_pt)

# 分类结果填充图 -- 测试集
plt.subplot(326)
plt.title('Contourf-Test')
# 绘制分类背景填充
plot_class_contourf(train_x, W)
# 绘制数据点
plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap=cm_pt)

plt.tight_layout()
plt.show()
