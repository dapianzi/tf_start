import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from utils import BATCH_SIZE, EPOCH, TEST_SET, TRAIN_SET, plot_history

np.set_printoptions(threshold=np.inf)

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()

x_train = x_train[:TRAIN_SET]
y_train = y_train[:TRAIN_SET]
x_test = x_test[:TEST_SET]
y_test = y_test[:TEST_SET]

x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape", x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("x_train.shape", x_train.shape)


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
print("""model total params should be :
Conv2D: %d,
BatchNormaliztion: %d,
Dense 1: %d,
Dense 2: %d
""" % (6 * 5 * 5 + 6, 6 * 4, 14 * 14 * 6 * 128 + 128, 128 * 10 + 10))

# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
plot_history(history)
