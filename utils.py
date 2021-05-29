import tensorflow as tf
from os import path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = path.dirname(__file__)
BATCH_SIZE = 32
EPOCH = 5
TRAIN_SET = 128
TEST_SET = 32


def get_data_set(dataset, cache_dir=""):
    """下载或从本地缓存读取数据集"""
    cache_dir = ROOT_DIR + "/cache" if cache_dir == "" else cache_dir
    if path.exists(cache_dir + "/datasets/" + dataset):
        return cache_dir + "/datasets/" + dataset
    else:
        url = "http://download.tensorflow.org/data/"
        return tf.keras.utils.get_file(dataset, url + dataset, cache_dir=cache_dir)


def get_arr_from_csv(dataset):
    path = get_data_set(dataset)
    df = pd.read_csv(path, header=0)
    nparr = np.array(df)
    return nparr


def plot_history(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(4, 6))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
