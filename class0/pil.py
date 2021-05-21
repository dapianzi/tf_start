from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

names = locals()

img0 = Image.open("./assets/pyCharm.png")
# print image info:
print(img0.size, img0.format, img0.mode, np.array(img0))
# save other format
# img0.save('./assets/pyCharm.tiff')
# img0.convert('RGB').save('./assets/pyCharm.jpeg')
# img0.convert('L').save('./assets/pyCharm.bmp')  # 灰度图

img1 = Image.open('./assets/pyCharm.tiff')  # 3通道图
img2 = Image.open('./assets/pyCharm.jpeg')
img3 = Image.open('./assets/pyCharm.bmp')
# 3通道图可以拆分
img4, img5, img6 = img2.split()
img7 = Image.merge('RGB', [img5, img6, img4])

plt.figure(figsize=(15, 15))

for i in range(8):
    plt.subplot(4, 3, i + 1)
    plt.axis('off')  # hide axis
    plt.imshow(names.get('img' + str(i)))
    plt.title(names.get('img' + str(i)).format)


# 去除 png 的白边
img_dir = '/Users/carl/Pictures/logos/'
logo = Image.open(img_dir + 'google.png')
# 将压缩的8位图像转成rgba
logo = logo.convert('RGBA')
# 分离通道
(logo_r, logo_g, logo_b, logo_a) = logo.split()
# 转换成numpy数组
arr_r = np.array(logo_r)
arr_g = np.array(logo_g)
arr_b = np.array(logo_b)
arr_a = np.array(logo_a)
# 筛选像素坐标
idx = (arr_r == 245) & (arr_g == 247) & (arr_b == 247)
# 修改为透明像素点
arr_r[idx] = 0
arr_g[idx] = 0
arr_b[idx] = 0
arr_a[idx] = 0
# 将numpy数组转回图片对象
shard_r = Image.fromarray(arr_r)
shard_g = Image.fromarray(arr_g)
shard_b = Image.fromarray(arr_b)
shard_a = Image.fromarray(arr_a)

rgb_dict = 'rgba'
for i in range(4):
    plt.subplot(4, 3, i+9)
    plt.axis('off')  # hide axis
    plt.imshow(names.get('shard_' + rgb_dict[i]))
    plt.title(names.get('shard_' + rgb_dict[i]).format)

# 合并通道，保存
Image.merge('RGBA', [shard_r, shard_g, shard_b, shard_a]).save(img_dir + 'logo-1.png', overWrite=True)

plt.tight_layout()
plt.show()