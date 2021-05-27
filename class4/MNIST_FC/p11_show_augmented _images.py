"""
数据增强 -- 图像变换
tensorflow.keras.preprocessing.image.ImageDataGenerator
"""
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 255,  # 缩放
    rotation_range=45,  # 旋转 0-45
    width_shift_range=.15,  # 宽度裁剪
    height_shift_range=.15,  # 高度裁剪
    horizontal_flip=False,  # 随机水平翻转
    zoom_range=0.5  # 缩放比例 (1-zoom_range, 1+zoom_range)
)

print("xtrain", x_train.shape)
x_train_subset1 = np.squeeze(x_train[:12])
print("xtrain_subset1", x_train_subset1.shape)
x_train_subset2 = x_train[:12]
print("xtrain_subset2", x_train_subset2.shape)

fig = plt.figure(figsize=(12, 2))
plt.set_cmap('gray')
# 显示原始图片
for i in range(0, len(x_train_subset1)):
    plt.subplot(2, 12, i + 1)
    plt.axis('off')
    plt.imshow(x_train_subset1[i])

# 显示增强后的图片
for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12, shuffle=False):
    print(x_batch.shape)
    for i in range(0, 12):
        plt.subplot(2, 12, i + 13)
        plt.axis('off')
        plt.imshow(np.squeeze(x_batch[i]))
    break

plt.tight_layout()
plt.show()
