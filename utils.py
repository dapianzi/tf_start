import tensorflow as tf
from os import path
import pandas as pd
import numpy as np

ROOT_DIR = path.dirname(__file__)


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
