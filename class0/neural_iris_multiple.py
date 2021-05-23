"""
多层神经网络进行鸢尾花分类
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

iris_train = utils.get_arr_from_csv('iris_training.csv')
iris_test = utils.get_arr_from_csv('iris_test.csv')


def pre_process(dataset):
    attrs, label = dataset[:, :4], dataset[:, 4]
    # 均值化
    attrs = attrs - np.mean(attrs)
    x = tf.cast(attrs, dtype=tf.float32)
    y = tf.one_hot(tf.constant(label, dtype=tf.int32), 3)
    return x, y, label


train_x, train_y, train_label = pre_process(iris_train)
test_x, test_y, test_label = pre_process(iris_test)

# 超参数
lr = 0.08
iters = 300
step = 30

train_accs = []
test_accs = []
train_losses = []
test_losses = []

W1 = tf.Variable(np.random.randn(4, 12), dtype=tf.float32)
W2 = tf.Variable(np.random.randn(12, 3), dtype=tf.float32)
B1 = tf.Variable(np.random.randn(12, ), dtype=tf.float32)
B2 = tf.Variable(np.random.randn(3, ), dtype=tf.float32)

print(train_x.dtype, W1.dtype)

for i in range(iters + 1):
    with tf.GradientTape() as tape:
        # 隐含层输出
        hidden_pred = tf.nn.relu(tf.matmul(train_x, W1) + B1)
        # 最终输出
        final_pred = tf.nn.softmax(tf.matmul(hidden_pred, W2) + B2)
        # 平均交叉熵作损失
        train_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=train_y, y_pred=final_pred))
    # test 正向结果
    test_pred = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(test_x, W1) + B1), W2) + B2
    )
    # test 损失
    test_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=test_y, y_pred=test_pred))
    # 训练集和测试集准确率
    train_acc = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(final_pred, axis=1), train_label
    ), tf.float32))
    test_acc = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(test_pred, axis=1), test_label
    ), tf.float32))

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    gd = tape.gradient(train_loss, [W1, B1, W2, B2])
    W1.assign_sub(lr*gd[0])
    B1.assign_sub(lr*gd[1])
    W2.assign_sub(lr*gd[2])
    B2.assign_sub(lr*gd[3])

    if i % step == 0:
        print("No. %i, Train Loss: %f, Train Acc: %f, Test Loss %f, Test Acc %f" % (
            i, train_loss, train_acc, test_loss, test_acc
        ))

plt.subplot(121)
plt.title('Losses')
plt.plot(train_losses, color='cyan', label='Train')
plt.plot(test_losses, color='orangered', label='Test')
plt.legend()
plt.subplot(122)
plt.title('Accuracies')
plt.plot(train_accs, color='cyan', label='Train')
plt.plot(test_accs, color='orangered', label='Test')
plt.legend()

plt.tight_layout()
plt.show()

plt.subplot
