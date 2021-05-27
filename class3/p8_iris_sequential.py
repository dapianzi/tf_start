import tensorflow as tf
import numpy as np
import utils

iris_train = utils.get_arr_from_csv('iris_training.csv')
iris_test = utils.get_arr_from_csv('iris_test.csv')
iris = np.concatenate((iris_train, iris_test), axis=0)
x_train, y_train = iris[:, :4], iris[:, 4]
print(x_train.shape, y_train.shape)

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([
    # softmax激活，l2正则化
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              # 标签y_是分类数值索引，预测值y是独热码概率分布
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()
