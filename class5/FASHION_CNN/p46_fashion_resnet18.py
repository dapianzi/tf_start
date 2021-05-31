import os

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

from utils import *

np.set_printoptions(threshold=np.inf)

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train = x_train[:TRAIN_SET]
y_train = y_train[:TRAIN_SET]
x_test = x_test[:TEST_SET]
y_test = y_test[:TEST_SET]
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape", x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("x_train.shape", x_train.shape)


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/ResNet18.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

print("""model total params should be :
###### 1 ######
3*3*64=%d
64*4=%d
out: 28*28*64
-----------------------------------------------------------
###### Block 2.1 ######
上轮输出深度d=64,本轮卷积核f=64
Conv2d 1: d*3*3*f
Conv2D 2: f*3*3*f
BN*2: f*4*2
sum=%d
上轮输出深度d=64,本轮卷积核f=64
Conv2d 1: d*3*3*f
Conv2d 2: f*3*3*f
BN*2: f2*4*2
sum=%d
out shape: [28,28,64]
-----------------------------------------------------------
###### Block 2.2 ######
上轮输出深度d=64,本轮卷积核f=128
Conv2d 1: d*3*3*f
Conv2D 2: f*3*3*f
Conv2D 3: d*1*1*f
BN*3: f*4*3
sum=%d
上轮输出深度d=128,本轮卷积核f=128
Conv2d 1: d*3*3*f
Conv2d 2: f*3*3*f
BN*2: f2*4*2
sum=%d
out shape: [14,14,128]
-----------------------------------------------------------
###### Block 2.3 ######
上轮输出深度d=128,本轮卷积核f=256
Conv2d 1: d*3*3*f
Conv2D 2: f*3*3*f
Conv2D 3: d*1*1*f
BN*3: f*4*3
sum=%d
上轮输出深度d=256,本轮卷积核f=256
Conv2d 1: d*3*3*f
Conv2d 2: f*3*3*f
BN*2: f2*4*2
sum=%d
out shape: [7,7,256]
-----------------------------------------------------------
###### Block 2.4 ######
上轮输出深度d=256,本轮卷积核f=512
Conv2d 1: d*3*3*f
Conv2D 2: f*3*3*f
Conv2D 3: d*1*1*f
BN*3: f*4*3
sum=%d
上轮输出深度d=512,本轮卷积核f=512
Conv2d 1: d*3*3*f
Conv2d 2: f*3*3*f
BN*2: f2*4*2
sum=%d
out shape: [4,4,512]
###### 2.total ######
total all: %d
-----------------------------------------------------------
###### 3 ######
Pooling out: [1,1,512]
Dense: 512*10+10=%d
""" % (
    3 * 3 * 64,
    64 * 4,
    64 * 64 * 9 + 9 * 64 * 64 + 8 * 64,
    64 * 64 * 9 + 64 * 64 * 9 + 64 * 8,
    128 * 64 * 10 + 9 * 128 * 128 + 12 * 128,
    128 * 128 * 9 + 128 * 128 * 9 + 128 * 8,
    256 * 128 * 10 + 9 * 256 * 256 + 12 * 256,
    256 * 256 * 9 + 256 * 256 * 9 + 256 * 8,
    512 * 256 * 10 + 9 * 512 * 512 + 12 * 512,
    512 * 512 * 9 + 512 * 512 * 9 + 512 * 8,
    np.sum(np.array([
        64 * 64 * 9 + 9 * 64 * 64 + 8 * 64,
        64 * 64 * 9 + 64 * 64 * 9 + 64 * 8,
        128 * 64 * 10 + 9 * 128 * 128 + 12 * 128,
        128 * 128 * 9 + 128 * 128 * 9 + 128 * 8,
        256 * 128 * 10 + 9 * 256 * 256 + 12 * 256,
        256 * 256 * 9 + 256 * 256 * 9 + 256 * 8,
        512 * 256 * 10 + 9 * 512 * 512 + 12 * 512,
        512 * 512 * 9 + 512 * 512 * 9 + 512 * 8,
    ])).item(),
    512 * 10 + 10,
)
      )
# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

############################################################################################    show   ############################################################################################

# 显示训练集和验证集的acc和loss曲线
plot_history(history)
