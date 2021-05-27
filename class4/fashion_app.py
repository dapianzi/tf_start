import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from libs.SequentialModel import SequentialFashion

types = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

ckpt_path = './checkpoint/fashion_aug.ckpt'
aug_path = './checkpoint/fashion_chpt.ckpt'
ori_path = './checkpoint/fashion_ori.ckpt'

model1 = SequentialFashion()
model2 = SequentialFashion()
model3 = SequentialFashion()

model1.load_weights(ori_path)
model2.load_weights(ckpt_path)
model3.load_weights(aug_path)

preNum = int(input("input the number of test pictures:"))


def read_img(name):
    path = 'FASHION_FC/' + name + '.jpeg'
    img = Image.open(path)
    image = plt.imread(path)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(image)
    # 压缩图片大小，适应模型输入
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_np = np.array(img.convert('L'))
    # 转成跟训练集一样的灰度图
    img_np = 255 - img_np
    plt.imshow(Image.fromarray(img_np))

    return img_np / 255.0


for i in range(preNum):
    name = input("the path of test picture:")
    img_arr = read_img(name)
    # 预测集是一个集合，需要在图片外面增加一个维度包裹
    x_predict = img_arr[tf.newaxis, ...]
    pred = (
        types[int(tf.argmax(model1.predict(x_predict), axis=1))],
        types[int(tf.argmax(model2.predict(x_predict), axis=1))],
        types[int(tf.argmax(model3.predict(x_predict), axis=1))]
    )

    tf.print(pred)
    print('\n')

    plt.pause(1)
    plt.close()
