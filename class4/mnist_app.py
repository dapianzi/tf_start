import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from libs.SequentialModel import SequentialMnist

ckpt_path = './checkpoint/mnist_aug.ckpt'
aug_path = './checkpoint/mnist_chpt.ckpt'
ori_path = './checkpoint/mnist_ori.ckpt'

model1 = SequentialMnist()
model2 = SequentialMnist()
model3 = SequentialMnist()

model1.load_weights(ori_path)
model2.load_weights(ckpt_path)
model3.load_weights(aug_path)

preNum = int(input("input the number of test pictures:"))


def read_img(name):
    path = 'MNIST_FC/' + name + '.png'
    img = Image.open(path)
    image = plt.imread(path)
    plt.set_cmap('gray')
    plt.imshow(image)
    # 压缩图片大小，适应模型输入
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_np = np.array(img.convert('L'))
    # 转成跟训练集一样的黑底白字
    # for i in range(28):
    #     for j in range(28):
    #         if img_np[i][j] < 200:
    #             img_np[i][j] = 255
    #         else:
    #             img_np[i][j] = 0

    img_np = np.where(img_np < 200, 255, 0)

    return img_np / 255.0


for i in range(preNum):
    name = input("the path of test picture:")
    img_arr = read_img(name)
    # 预测集是一个集合，需要在图片外面增加一个维度包裹
    x_predict = img_arr[tf.newaxis, ...]
    pred = (
        tf.argmax(model1.predict(x_predict), axis=1),
        tf.argmax(model2.predict(x_predict), axis=1),
        tf.argmax(model3.predict(x_predict), axis=1)
    )

    tf.print(pred)
    print('\n')

    plt.pause(1)
    plt.close()
