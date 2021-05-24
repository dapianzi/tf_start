import tensorflow as tf

features = tf.constant([[12, 3], [23, 7], [10, 1], [17, 5]])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)
