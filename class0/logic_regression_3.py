"""
N元逻辑回归2分类
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

# 预置参数
plt.rcParams['font.sans-serif'] = "Arial Unicode MS"
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.figure(figsize=(8, 4))

cm_pt = mpl.colors.ListedColormap(['royalblue', 'orangered'])
cm_bg = mpl.colors.ListedColormap(['cyan', 'gold'])

iris_train = utils.get_arr_from_csv('iris_training.csv')
iris_test = utils.get_arr_from_csv('iris_test.csv')


# 处理数据集
def pre_precess(dataset):
    x = dataset[:, :4]  # 提取前2列特征
    y = dataset[:, 4]  # 提取标签
    # 过滤 label=2 的数据
    x = x[y < 2]
    y = y[y < 2]
    # 均值化
    x = x - np.mean(x, axis=0)
    return x, y


train_x, train_y = pre_precess(iris_train)
test_x, test_y = pre_precess(iris_test)
n1 = len(train_x)
n2 = len(test_x)
print(n1, n2)

lr = 0.01
iters = 500
step = 50

train_X = tf.cast(tf.concat((np.ones(n1, ).reshape(-1, 1), train_x), axis=1), tf.float32)
test_X = tf.cast(tf.concat((np.ones(n2, ).reshape(-1, 1), test_x), axis=1), tf.float32)
train_Y = tf.cast(train_y.reshape(-1, 1), tf.float32)
test_Y = tf.cast(test_y.reshape(-1, 1), tf.float32)
print(train_X.shape, train_Y.shape)

W = tf.Variable(np.random.randn(5, 1), dtype=tf.float32)

train_losses = []
train_acces = []
test_losses = []
test_acces = []

for i in range(iters + 1):
    with tf.GradientTape() as tape:
        PRED_TRAIN = 1 / (1 + tf.exp(-tf.matmul(train_X, W)))
        LOSS_TRAIN = -tf.reduce_mean(
            train_Y * tf.math.log(PRED_TRAIN) + (1 - train_Y) * tf.math.log(1 - PRED_TRAIN)
        )
    PRED_TEST = 1 / (1 + tf.exp(-tf.matmul(test_X, W)))
    LOSS_TEST = -tf.reduce_mean(
        test_Y * tf.math.log(PRED_TEST) + (1 - test_Y) * tf.math.log(1 - PRED_TEST)
    )

    train_acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.where(PRED_TRAIN > 0.5, 1., 0.), train_Y), tf.float32)
    )
    test_acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.where(PRED_TEST > 0.5, 1., 0.), test_Y), tf.float32)
    )

    train_losses.append(LOSS_TRAIN)
    test_losses.append(LOSS_TEST)
    train_acces.append(train_acc)
    test_acces.append(test_acc)

    dW = tape.gradient(LOSS_TRAIN, W)
    W.assign_sub(dW * lr)

    if i % step == 0:
        print("第%d轮：Loss: %f, Acc: %fs"% (i, LOSS_TRAIN, train_acc))

plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(train_losses, linewidth=2, color='cyan', label="Train")
plt.plot(test_losses, linewidth=1.5, color='orangered', label="Test")
plt.legend()

plt.subplot(1,2,2)
plt.title('Accuracy')
plt.plot(train_acces, linewidth=2, color='cyan', label="Train")
plt.plot(test_acces, linewidth=1.5, color='orangered', label="Test")
plt.legend()

plt.tight_layout()
plt.show()