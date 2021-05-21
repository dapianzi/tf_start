from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_set, train_label), (test_set, test_label) = mnist.load_data()

print(train_set.shape, train_set.dtype, test_set.shape, test_set.dtype)

plt.figure(figsize=(6, 6))
for i in range(9):
    n = np.random.randint(0, 60000)
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.imshow(train_set[n], cmap='gray')
    plt.title(train_label[n])

plt.tight_layout()
plt.show()
