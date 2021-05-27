"""
训练数据增强的图片
"""
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from libs.SequentialModel import SequentialMnist

np.set_printoptions(threshold=np.inf)

# load data
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度,从(60000, 28, 28)reshape为(60000, 28, 28, 1)

# image augment
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)

# load model
model = SequentialMnist()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

save_path = './checkpoint/fashion_aug.ckpt'

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)  # 只保存最好的模型参数

if os.path.exists(save_path + '.index'):
    model.load_weights(filepath=save_path)
else:
    # training
    history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=10,
                        validation_data=(x_test, y_test),
                        validation_freq=1, callbacks=[cp_callback])

    # plot
    plt.figure(figsize=(4, 6))

    plt.subplot(211)
    plt.title('Acc')
    plt.plot(history.history['sparse_categorical_accuracy'], label="Train")
    plt.plot(history.history['val_sparse_categorical_accuracy'], label="Test")
    plt.legend()
    plt.subplot(212)
    plt.title('Loss')
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Test")
    plt.legend()

    plt.tight_layout()
    plt.show()

print(model.trainable_variables)
file = open('./fashion_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
model.summary()
