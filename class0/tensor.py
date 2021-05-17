import tensorflow as tf
import numpy as np

# 直接创建
a = tf.constant([1., 2.], dtype=tf.float32)

# 从 numpy 创建
b = tf.constant(np.array([1., 2.]))
print(a, b)

# 一般来说float32满足精度需求，而且在GPU可以获得比较理想的速度
c = tf.cast(b, dtype=tf.float32)
print(c)

na = np.arange(12).reshape(3, 4)
print(tf.is_tensor(na), isinstance(na, tf.Tensor))
ta = tf.convert_to_tensor(na)
print(ta)

# specified
t1 = tf.ones((1, 2, 3), tf.int32, name='t1')
t2 = tf.zeros((2, 3), dtype=tf.float32, name='t2')
t3 = tf.fill((2, 3), 6)
t4 = tf.fill((2, 3), 6.0)
t5 = tf.constant(9, shape=(2, 3))
print(t1, t2, t3.dtype, t4.ndim, t5)

# random
n1 = tf.random.normal((3, 4), mean=0, stddev=1., dtype=tf.float32)
n2 = tf.random.truncated_normal((1, 2))
n3 = tf.random.uniform((2, 3))

# attributes
print(tf.shape(n1), tf.rank(n2), tf.size(n3))

# transform
n4 = tf.convert_to_tensor(np.arange(24).reshape((2,3,4)))
n5 = tf.transpose(n4, (2, 0, 1))
print(n4, n5)
