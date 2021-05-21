"""
一元逻辑回归
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

housing = tf.keras.datasets.boston_housing
(props, prices), _ = housing.load_data()

rm = props[:, 5]
print(np.mean(rm), np.max(rm), np.min(rm))
labels = tf.where(rm > np.random.normal(6, 0.1, (len(rm),)), 1., 0.)
train_x = rm - np.mean(rm)  # like tf.reduce_mean

learn_rate = 0.08
iter_times = 1000
display_step = 100

cross_train = []  # 训练交叉熵
acc_train = []  # 训练准确率

w = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('data & predict')
plt.scatter(train_x, labels)
X_ = np.arange(np.floor(np.min(train_x)), np.ceil(np.max(train_x)), 0.1)
Y_ = 1 / (1 + tf.exp(-(w * X_ + b)))
plt.plot(X_, Y_)

for i in range(iter_times):
    with tf.GradientTape() as tape:
        pred_train = 1 / (1 + tf.exp(-(w * train_x + b)))
        # 交叉熵的计算
        loss = -tf.reduce_mean(labels * tf.math.log(pred_train) + (1. - labels) * tf.math.log(1. - pred_train))
        # 成功率：
        # 1. 对pred_train进行0|1转换
        # 2. 对比预测结果和实际标签
        # 3. 将True|False转换成float32
        # 4. 计算平均值 === 计算成功率
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train > 0.5, 1., 0.), labels), tf.float32))

    # 保存每一轮的loss和acc
    cross_train.append(loss)
    acc_train.append(acc)

    # 后向传播更新参数
    dw, db = tape.gradient(loss, [w, b])  # 以loss作为函数对[w,b]求偏导数（梯度）
    w.assign_sub(dw * learn_rate)
    b.assign_sub(db * learn_rate)

    if i % display_step == 0:
        print(f"第{i}轮: Train Loss: {loss}, Accuracy: {acc}")
        Y_ = 1 / (1 + tf.exp(-(w * X_ + b)))
        plt.plot(X_, Y_)

plt.subplot(1, 2, 2)
plt.title('loss & acc')
plt.plot(cross_train, label="Loss[Cross]")
plt.plot(acc_train, label="Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
