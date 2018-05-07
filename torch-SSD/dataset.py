import random
import os
from copy import deepcopy
import math

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transform

import tools

on_server = False


img_size = 300  # 规定每一张图片的大小都是300*300


class VOC2007(Dataset):
    """

    数据预处理只进行了图片缩放和水平随机翻转
    其他的操作过于复杂，所以直接放弃
    """
    def __init__(self, img_path, txt_path, transform, whether_train):
        """

        :param img_path: 图片存放的文件夹路径
        :param txt_path: txt文件的路径
        :param transform: 将图片转化为tensor并进行归一化处理
        :param whether_train: 是否处于训练模式
        """
        self.transform = transform
        self.whether_train = whether_train

        with open(txt_path, 'r') as f:

            lines = f.readlines()

            # 获取图片列表
            self.img_list = [
                os.path.join(img_path, line.split(',')[0]) for line in lines
            ]

            # 获取标签列表
            self.label_list = []

            for line in lines:

                """
                
                这里的每一个element代表着列表中的一个元素
                一个元素是一个list，列表长度代表着一张图片中的物体个数
                list中的每一个元素是一个dict，每一个dict有两个key，第一个是class，第二个是bbox
                """
                line = line[0:-1]  # 删除最后一个多余的换行符
                element = []

                width = int(line.split(',')[1])  # 图片的width
                height = int(line.split(',')[2])  # 图片的height

                for item in line.split(',')[3:]:
                    temp_list = item[1:-1].split()

                    temp_dict = {}
                    temp_dict['class'] = int(temp_list[0])
                    temp_dict['bbox'] = list(map(int, temp_list[1:]))

                    xmin = temp_dict['bbox'][0]
                    ymin = temp_dict['bbox'][1]
                    xmax = temp_dict['bbox'][2]
                    ymax = temp_dict['bbox'][3]

                    temp_dict['bbox'][0] = (xmin + xmax) / 2 / width  # 中心点x相对坐标
                    temp_dict['bbox'][1] = (ymin + ymax) / 2 / height  # 中心点y相对坐标
                    temp_dict['bbox'][2] = (xmax - xmin) / width  # bbox的相对width
                    temp_dict['bbox'][3] = (ymax - ymin) / height  # bbox的相对height

                    element.append(temp_dict)

                self.label_list.append(element)

    def __getitem__(self, index):

        # 获取图片
        img_path = self.img_list[index]
        raw_img = Image.open(img_path)

        # 获取标签，这里需要使用deepcopy，否则将会改变原始的label list !!!
        label = deepcopy(self.label_list[index])

        # 将图片缩放到300*300大小
        img = raw_img.resize((img_size, img_size), Image.ANTIALIAS)

        # 如果在训练模式下，将以0.5的概率翻转图片
        if self.whether_train is True and random.uniform(0, 1) > 0.5:
            img, label = self.hflip(img, label)

        # 将图片转化为tensor，并进行归一化
        img = self.transform(img)

        if self.whether_train is False:
            return raw_img, torch.unsqueeze(img, dim=0), label

        else:
            tar_loc = torch.zeros((8732, 4))
            tar_conf = torch.zeros(8732).type(torch.LongTensor)

            for item in label:
                for index, default_box in enumerate(tools.default_boxs):

                    # 如果IOU系数大于或等于0.5，就认为匹配
                    if tools.iou(default_box, item['bbox']) >= 0.5:
                        tar_conf[index] = item['class']

                        tar_loc[index][0] = (item['bbox'][0] - default_box[0]) / default_box[2]
                        tar_loc[index][1] = (item['bbox'][1] - default_box[1]) / default_box[3]
                        tar_loc[index][2] = math.log(item['bbox'][2] / default_box[2])
                        tar_loc[index][3] = math.log(item['bbox'][3] / default_box[3])

                        tar_loc[index][0] /= 0.1
                        tar_loc[index][1] /= 0.1
                        tar_loc[index][2] /= 0.2
                        tar_loc[index][3] /= 0.2

            return img, tar_conf, tar_loc

    def __len__(self):
        return len(self.img_list)

    def hflip(self, image, label):

        # 改变图片
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 改变标签
        for index in range(len(label)):

            label[index]['bbox'][0] = 1 - label[index]['bbox'][0]

        return image, label


# 定义文件路径
train_img_dir = '/home/zcy/Desktop/VOC/VOC2007/trainval/JPEGImages/' \
    if on_server is False else '/home/chaoyiz/VOC/train/JPEGImages/'
train_txt_dir = './txt/train.txt'

test_img_dir = '/home/zcy/Desktop/VOC/VOC2007/test/JPEGImages/' \
    if on_server is False else '/home/chaoyiz/VOC/test/JPEGImages/'

test_txt_dir = './txt/test.txt'

# 看github上大部分代码使用的均值和方差
mean = [104 / 255, 117 / 255, 123 / 255]
std = [1, 1, 1]

# # 使用voc2007数据集计算出来的均值和方差
# mean = [0.4485, 0.4250, 0.3920]
# std = [0.2720, 0.2687, 0.2814]

data_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean, std)
])

# 定义数据集
train_ds = VOC2007(train_img_dir, train_txt_dir, data_transform, True)
test_ds = VOC2007(test_img_dir, test_txt_dir, data_transform, False)


# # 测试代码：使用 DataLoader
# from torch.utils.data import DataLoader
# train_dl = DataLoader(train_ds, 32, True, num_workers=2, pin_memory=True)
# for index, (img, tar_conf, tar_loc) in enumerate(train_dl):
#     print(index, img.size(), tar_conf.size(), tar_loc.size())
