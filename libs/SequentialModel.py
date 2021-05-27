from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class SequentialIris(Model):
    """
    简单的鸢尾花分类训练模型
    """
    def __init__(self):
        super(SequentialIris, self).__init__()
        # 一层softmax激活网络，应用L2正则化
        self.d1 = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2())

    def call(self, x):
        y = self.d1(x)
        return y


class SequentialMnist(Model):
    """
    简单的mnist分类训练模型
    """
    def __init__(self):
        super(SequentialMnist, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        h1 = self.d1(x)
        y = self.d2(h1)
        return y


class SequentialFashion(Model):
    """
    简单的fashion分类训练模型
    """
    def __init__(self):
        super(SequentialFashion, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        h1 = self.d1(x)
        y = self.d2(h1)
        return y

