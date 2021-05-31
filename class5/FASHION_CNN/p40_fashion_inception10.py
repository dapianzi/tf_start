import os

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dense, \
    GlobalAveragePooling2D

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


class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x,
                       training=False)  # 在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。推理时 training=False效果好
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        # !! concat, not add
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = Inception10(num_blocks=2, num_classes=10)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Inception10.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

print("""model total params should be :
###### 1 ######
Conv2d: 3*3*16+16=%d
BN: 16*4=%d
out: 28*28*16
-----------------------------------------------------------
###### 2*2 ######
###### 2.1 InceptionBlk strides=2 #####
in : [28, 28, 16], filters=16
Conv2D 1: 16*1*1*16+16=%d
Conv2D 2_1: 16*1*1*16+16=%d
Conv2D 2_2: 16*3*3*16+16=%d
Conv2D 3_1: 16*1*1*16+16=%d
Conv2D 3_2: 16*5*5*16+16=%d
MaxPooling : [28, 28, 16] 
Conv2D 4: 16*1*1*16+16=%d
BN*6: 16*4*6=%d 
out : [14, 14, 64] (with concat)
-----------------------------------------------------------
###### 2.2 InceptionBlk strides=1 ######
in: [14, 14, 64], filters=16
Conv2D 1: 64*1*1*16+16=%d
Conv2D 2_1: 64*1*1*16+16=%d
Conv2D 2_2: 16*3*3*16+16=%d
Conv2D 3_1: 64*1*1*16+16=%d
Conv2D 3_2: 16*5*5*16+16=%d
MaxPooling : [28, 28, 16] 
Conv2D 4: 64*1*1*16+16=%d 
BN*6: 16*4*6=%d
out : [14, 14, 64]
-----------------------------------------------------------
###### 2.3 InceptionBlk strides=2 ######
in: [14, 14, 64], filters=32
Conv2D 1: 64*1*1*32+32=%d
Conv2D 2_1: 64*1*1*32+32=%d
Conv2D 2_2: 32*3*3*32+32=%d
Conv2D 3_1: 64*1*1*32+32=%d
Conv2D 3_2: 32*5*5*32+32=%d
MaxPooling : [7, 7, 32] 
Conv2D 4: 64*1*1*32+32=%d 
BN*6: 32*4*6=%d
out : [7, 7, 128]
-----------------------------------------------------------
###### 2.4 InceptionBlk strides=1 ######
in: [7, 7, 128], filters=32
Conv2D 1: 128*1*1*32+32=%d
Conv2D 2_1: 128*1*1*32+32=%d
Conv2D 2_2: 32*3*3*32+32=%d
Conv2D 3_1: 128*1*1*32+32=%d
Conv2D 3_2: 32*5*5*32+32=%d
MaxPooling : [7, 7, 32] 
Conv2D 4: 128*1*1*32+32=%d 
BN*6: 32*4*6=%d
out : [7, 7, 128]
-----------------------------------------------------------
total: %d
-----------------------------------------------------------
###### 3. Dense ######
in: [7, 7, 128]
AvgPooling: [1, 1, 128]
Dense: 128*10+10=%d
""" % (
    3 * 3 * 16 + 16,
    16 * 4,
    16 * 1 * 1 * 16 + 16,
    16 * 1 * 1 * 16 + 16,
    16 * 3 * 3 * 16 + 16,
    16 * 1 * 1 * 16 + 16,
    16 * 5 * 5 * 16 + 16,
    16 * 1 * 1 * 16 + 16,
    16 * 4 * 6,
    64 * 1 * 1 * 16 + 16,
    64 * 1 * 1 * 16 + 16,
    16 * 3 * 3 * 16 + 16,
    64 * 1 * 1 * 16 + 16,
    16 * 5 * 5 * 16 + 16,
    64 * 1 * 1 * 16 + 16,
    16 * 4 * 6,
    64 * 1 * 1 * 32 + 32,
    64 * 1 * 1 * 32 + 32,
    32 * 3 * 3 * 32 + 32,
    64 * 1 * 1 * 32 + 32,
    32 * 5 * 5 * 32 + 32,
    64 * 1 * 1 * 32 + 32,
    32 * 4 * 6,
    128 * 1 * 1 * 32 + 32,
    128 * 1 * 1 * 32 + 32,
    32 * 3 * 3 * 32 + 32,
    128 * 1 * 1 * 32 + 32,
    32 * 5 * 5 * 32 + 32,
    128 * 1 * 1 * 32 + 32,
    32 * 4 * 6,
    np.sum(np.array([
        16 * 1 * 1 * 16 + 16,
        16 * 1 * 1 * 16 + 16,
        16 * 3 * 3 * 16 + 16,
        16 * 1 * 1 * 16 + 16,
        16 * 5 * 5 * 16 + 16,
        16 * 1 * 1 * 16 + 16,
        16 * 4 * 6,
        64 * 1 * 1 * 16 + 16,
        64 * 1 * 1 * 16 + 16,
        16 * 3 * 3 * 16 + 16,
        64 * 1 * 1 * 16 + 16,
        16 * 5 * 5 * 16 + 16,
        64 * 1 * 1 * 16 + 16,
        16 * 4 * 6,
        64 * 1 * 1 * 32 + 32,
        64 * 1 * 1 * 32 + 32,
        32 * 3 * 3 * 32 + 32,
        64 * 1 * 1 * 32 + 32,
        32 * 5 * 5 * 32 + 32,
        64 * 1 * 1 * 32 + 32,
        32 * 4 * 6,
        128 * 1 * 1 * 32 + 32,
        128 * 1 * 1 * 32 + 32,
        32 * 3 * 3 * 32 + 32,
        128 * 1 * 1 * 32 + 32,
        32 * 5 * 5 * 32 + 32,
        128 * 1 * 1 * 32 + 32,
        32 * 4 * 6
    ])).item(),
    128 * 10 + 10
)
      )
# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
plot_history(history)
