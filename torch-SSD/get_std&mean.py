import os
from PIL import Image

import torch
import torchvision.transforms as tran

voc2007 = '/home/zcy/Desktop/VOC/VOC2007/trainval/JPEGImages/'

img_size = 300

r_mean = 0
g_mean = 0
b_mean = 0

r_std = 0
g_std = 0
b_std = 0

pix_num = 0

trans = tran.ToTensor()

for index, img in enumerate(os.listdir(voc2007)):
    img_path = os.path.join(voc2007, img)
    img = Image.open(img_path)

    img = img.resize((img_size, img_size), Image.ANTIALIAS)

    width, height = img.size
    pix_num += (width * height)

    img = trans(img)

    r_mean += img[0, :, :].sum()
    g_mean += img[1, :, :].sum()
    b_mean += img[2, :, :].sum()

    print(index)

r_mean /= pix_num
g_mean /= pix_num
b_mean /= pix_num

for index, img in enumerate(os.listdir(voc2007)):
    img_path = os.path.join(voc2007, img)
    img = Image.open(img_path)

    img = img.resize((img_size, img_size), Image.ANTIALIAS)

    img = trans(img)

    r_std += (img[0, :, :] - r_mean).__pow__(2).sum()
    g_std += (img[1, :, :] - g_mean).__pow__(2).sum()
    b_std += (img[2, :, :] - b_mean).__pow__(2).sum()

    print(index)

print('mean:')
print(r_mean)
print(g_mean)
print(b_mean)

print('std:')
print(torch.sqrt(r_std / pix_num))
print(torch.sqrt(g_std / pix_num))
print(torch.sqrt(b_std / pix_num))
