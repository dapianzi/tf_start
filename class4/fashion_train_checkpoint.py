"""
断点续训，保存模型
"""
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from libs.SequentialModel import SequentialFashion

# load data
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_test.shape)

# get model
model = SequentialFashion()
# compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# load weights
chpt_path = "./checkpoint/fashion_chpt.ckpt"
save_path = "./checkpoint/fashion_ori.ckpt"
if os.path.exists(save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(save_path)
    save_path = chpt_path

# train with checkpoint
# see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)  # 只保存最好的模型参数

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
# summary
model.summary()

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
