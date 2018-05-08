"""将voc数据集信息统一整理到一个txt文件下"""

import os
import xml.etree.ElementTree as ET

# VOC 20类数据
voc = {
    'background': 0,  # 这里背景规定为第0类
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20,
}

train07_xml_dir = '/home/zcy/Desktop/dataset/VOC/VOC2007/trainval/Annotations/'

train12_xml_dir = '/home/zcy/Desktop/dataset/VOC/VOC2012/trainval/Annotations/'

test_xml_dir = '/home/zcy/Desktop/dataset/VOC/VOC2007/test/Annotations/'

os.mkdir('./txt/')
train_txt_dir = './txt/train.txt'
test_txt_dir = './txt/test.txt'


# 获取voc2007训练集txt文件
for item in os.listdir(train07_xml_dir):

    tree = ET.parse(os.path.join(train07_xml_dir, item))
    root = tree.getroot()

    lines = ''  # 待写入txt文件的一行信息

    filename = root[1].text  # filename
    width = root[4][0].text  # image width
    height = root[4][1].text  # image height

    lines += str(filename + ',')
    lines += str(width + ',')
    lines += str(height + ',')

    for obj in root.iter('object'):  # 遍历图片中的每一个物体

        cat = obj[0].text  # object class

        bbox = obj.find('bndbox')

        xmin = ' ' + bbox[0].text + ' '
        ymin = bbox[1].text + ' '
        xmax = bbox[2].text + ' '
        ymax = bbox[3].text

        lines += '[' + str(voc[cat]) + xmin + ymin + xmax + ymax + '],'

    lines = lines[0:-1]  # 删除一行中最后一个多余的逗号

    with open(train_txt_dir, 'a') as f:
        f.writelines(lines + '\n')

# 获取VOC2007测试集txt文件
for item in os.listdir(test_xml_dir):

    tree = ET.parse(test_xml_dir + item)
    root = tree.getroot()

    lines = ''  # 待写入txt文件的一行信息

    filename = root[1].text  # filename
    width = root[4][0].text  # image width
    height = root[4][1].text  # image height

    lines += str(filename + ',')
    lines += str(width + ',')
    lines += str(height + ',')

    for obj in root.iter('object'):  # 遍历图片中的每一个物体

        cat = obj[0].text  # object class

        bbox = obj.find('bndbox')

        xmin = ' ' + bbox[0].text + ' '
        ymin = bbox[1].text + ' '
        xmax = bbox[2].text + ' '
        ymax = bbox[3].text

        lines += '[' + str(voc[cat]) + xmin + ymin + xmax + ymax + '],'

    lines = lines[0:-1]  # 删除一行中最后一个多余的逗号

    with open(test_txt_dir, 'a') as f:
        f.writelines(lines + '\n')

# 获取VOC2012训练集txt文件
for item in os.listdir(train12_xml_dir):

    tree = ET.parse(os.path.join(train12_xml_dir, item))
    root = tree.getroot()

    lines = ''  # 待写入txt文件的一行信息

    filename = root.find('filename').text  # filename

    size = root.find('size')
    width = size.find('width').text  # image width
    height = size.find('height').text  # image height

    lines += str(filename + ',')
    lines += str(width + ',')
    lines += str(height + ',')

    for obj in root.iter('object'):  # 遍历图片中的每一个物体

        cat = obj[0].text  # object class

        bbox = obj.find('bndbox')

        xmin = ' ' + bbox.find('xmin').text + ' '
        ymin = bbox.find('ymin').text + ' '
        xmax = bbox.find('xmax').text + ' '
        ymax = bbox.find('ymax').text

        lines += '[' + str(voc[cat]) + xmin + ymin + xmax + ymax + '],'

    lines = lines[0:-1]  # 删除一行中最后一个多余的逗号

    with open(train_txt_dir, 'a') as f:
        f.writelines(lines + '\n')
