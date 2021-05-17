import numpy as np
import tensorflow as tf

a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
print(tf.where(tf.greater(a, b), a, b))

x = np.random.RandomState().rand()
rdm = np.random.RandomState(seed=512)
m = rdm.rand()
n = rdm.rand(2, 3)
print(x, m, n)
print(np.random.RandomState(seed=512).rand())

# vstack
print(np.vstack((a, b)))

# mgrid -- 矩阵乘
print(np.mgrid[1:4:1])
x, y = np.mgrid[1:4:1, 2:3:.5]
print(x, y)
# ravel np.c_[]
print(np.c_[x.ravel(), y.ravel()]) # 快速生成张量切片的下标索引

