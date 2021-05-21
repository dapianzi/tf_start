"""
单层神经网络解决鸢尾花分类问题
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

# 下载数据集
iris_train = utils.get_arr_from_csv('iris_training.csv')
iris_test = utils.get_arr_from_csv('iris_test.csv')
# 提取特征和标签
train_attrs = iris_train[:, :4]
train_label = iris_train[:, 4]
print(train_attrs.shape, train_label.shape)
test_attrs = iris_test[:, :4]
test_label = iris_test[:, 4]
# 标准化, 数量级相同因此不需要归一化
train_attrs = train_attrs - np.mean(train_attrs, axis=0)
test_attrs = test_attrs - np.mean(test_attrs, axis=0)
# 转为float32
train_attrs = tf.cast(train_attrs, dtype=tf.float32)
test_attrs = tf.cast(test_attrs, dtype=tf.float32)
# 独热码
train_onehot = tf.one_hot(tf.constant(train_label, dtype=tf.int32), 3)
test_onehot = tf.one_hot(tf.constant(test_label, dtype=tf.int32), 3)
# 超参数
lr = 0.1
epoch = 1000
step = 50
# 初始化 W 和 B ，参与预测的是4个属性，输出的是3个分类，因此转换矩阵是（4，3）
W = tf.Variable(np.random.randn(4, 3), dtype=tf.float32)
# （3，）维向量作为偏移量
B = tf.Variable(np.zeros(3), dtype=tf.float32)
# 训练过程记录
acc_train = []
acc_test = []
ace_train = []
ace_test = []

for i in range(epoch + 1):
    with tf.GradientTape() as tape:
        # use softmax， 激活函数
        PRED = tf.nn.softmax(tf.matmul(train_attrs, W) + B)
        # average of cce， 平均分类交叉熵
        curr_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=train_onehot, y_pred=PRED))
    # 测试集不需要计算梯度
    PRED_TEST = tf.nn.softmax(tf.matmul(test_attrs, W) + B)
    curr_loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=test_onehot, y_pred=PRED_TEST))
    # 计算准确率：
    # 1. PRED结果为（N, 3）矩阵，提取每一行最大值的索引（最可能的分类）
    most_pro = tf.argmax(PRED.numpy(), axis=1)
    # 2. 索引值正好就是转为独热码之前的标签值 train_label，作一一对比
    compare = tf.equal(most_pro, train_label)
    # 3. 将 (True,False) 转为 (1.,0.)，计算平均值（arg平均值 = count(1)+count(0)/total = count(1)/total = acc准确率）
    curr_acc_train = tf.reduce_mean(tf.cast(compare, dtype=tf.float32))  # 要计算平均值，所以是float32
    # 保存训练参数
    acc_train.append(curr_acc_train)
    curr_acc_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_TEST.numpy(), axis=1), test_label), tf.float32))
    acc_test.append(curr_acc_test)
    ace_train.append(curr_loss)
    ace_test.append(curr_loss_test)
    # 计算梯度，更新模型参数
    grads = tape.gradient(curr_loss, [W, B])
    W.assign_sub(lr * grads[0])
    B.assign_sub(lr * grads[1])

    if i % step == 0:
        print("No.%i epoch, Train Acc: %f, Train Loss: %f, Test Acc: %f, Test Loss: %f" % (
            i, curr_acc_train, curr_loss, curr_acc_test, curr_loss_test))

# 绘制准确率
plt.subplot(1, 2, 1)
plt.plot(acc_train, color='lightgreen', label="Train Accuracy")
plt.plot(acc_test, color='orangered', label="Test Accuracy")
plt.legend()
# 绘制损失
plt.subplot(1, 2, 2)
plt.plot(ace_train, color='lightgreen', label="Train Average CCE")
plt.plot(ace_test, color='orangered', label="Test Average CCE")
plt.legend()

plt.tight_layout()
plt.show()