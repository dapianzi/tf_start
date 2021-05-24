import tensorflow as tf
import numpy as np


with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant([np.pi/6, np.pi/3, np.pi/2, np.pi], dtype=tf.float64))
    y = tf.sin(x)
grad = tape.gradient(y, x)
print(grad)
