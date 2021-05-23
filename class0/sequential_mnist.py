"""
使用Sequential创建模型训练mnist识别
"""
import os.path as path
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIN_FLAG = False
EPOCHS = 10
BATCH_SIZE = 128

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
print(train_x.shape, test_x.shape)
print(train_y[0])


def normalize(arr):
    # 归一化
    return tf.cast(arr / 255, tf.float32)


train_X, test_X = normalize(train_x), normalize(test_x)
train_Y, test_Y = tf.cast(train_y, dtype=tf.int32), tf.cast(test_y, dtype=tf.int32)
# 定义Sequential 模型各层
model = tf.keras.Sequential(name='mnist_1')
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 非卷积，输入（784，）一维向量
model.add(tf.keras.layers.Dense(128, activation='relu'))  # relu激活，输出（128，）
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # softmax激活，输出（10，）

# print(model.summary())
# 设置 loss 和 metrics
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")

weights_path = "./assets/mnist_sequential.h5"
if TRAIN_FLAG or not path.exists(weights_path):
    model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    model.save_weights(filepath=weights_path, overwrite=True)
else:
    model.load_weights(filepath=weights_path)

model.evaluate(test_X, test_Y, verbose=2)

n = 12
plt.figure()
rand_idx = np.random.randint(0, len(test_X) - n)
test_data = test_X[rand_idx:(rand_idx + n)]
pred = np.argmax(model.predict([test_data]), axis=1)
print(pred)
# draw imgs and preds
for i in range(n):
    plt.subplot(3, 4, 1 + i)
    plt.axis('off')
    plt.title(pred[i])
    plt.imshow(test_data[i], cmap='gray')

plt.tight_layout()
plt.show()
