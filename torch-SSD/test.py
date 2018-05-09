"""可视化目标探测效果"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn.functional as F

from dataset import test_ds

from ssd300 import SSD300

import tools

# 设置默认的tensor类型
torch.set_default_tensor_type('torch.cuda.FloatTensor')

test_img_index = 113
conf_threshold = 0.2
nms_threshold = 0.45

test_img, norm_img, label = test_ds.__getitem__(test_img_index)

width, height = test_img.size

# 实例化一个网络模型
net = torch.nn.DataParallel(SSD300())

# 加载参数
net.load_state_dict(torch.load('./net5-5.378.pth'))


# pred_conf: 预测得到的分类概率向量---1, 8732, 21
# pred_loc: 预测得到的偏移向量---1, 8732, 4
norm_img = norm_img.cuda()
pred_conf, pred_loc = net(norm_img)

pred_loc = torch.squeeze(pred_loc)
pred_conf = torch.squeeze(pred_conf)

# 将pred_conf转换为概率向量
pred_conf = F.softmax(pred_conf, dim=1)

# 将预测得到的loc变换为真实的loc
pred_loc[:, 0] *= 0.1
pred_loc[:, 1] *= 0.1
pred_loc[:, 2] *= 0.2
pred_loc[:, 3] *= 0.2

default_box = torch.Tensor(tools.default_boxs)

pred_loc[:, 0] = pred_loc[:, 0] * default_box[:, 2] + default_box[:, 0]
pred_loc[:, 1] = pred_loc[:, 1] * default_box[:, 3] + default_box[:, 1]
pred_loc[:, 2] = torch.exp(pred_loc[:, 2]) * default_box[:, 2]
pred_loc[:, 3] = torch.exp(pred_loc[:, 3]) * default_box[:, 3]


# 选取置信度大于阈值，并且所属类别不是背景的bbox
mask = (torch.max(pred_conf, dim=1)[0] >= conf_threshold) * (torch.max(pred_conf, dim=1)[1] > 0)

print('before nms {} objects left'.format(int(mask.type(torch.FloatTensor).sum().item())))

conf_mask = mask.unsqueeze(1).expand_as(pred_conf)
pred_conf = pred_conf[conf_mask].view(-1, 21)

loc_mask = mask.unsqueeze(1).expand_as(pred_loc)
pred_loc = pred_loc[loc_mask].view(-1, 4)


# 创建颜色映射表
color_map = {
    1: 'peru',
    2: 'brown',
    3: 'orange',
    4: 'fuchsia',
    5: 'tomato',
    6: 'yellow',
    7: 'deeppink',
    8: 'red',
    9: 'darkviolet',
    10: 'skyblue',
    11: 'cyan',
    12: 'lawngreen',
    13: 'teal',
    14: 'wheat',
    15: 'sienna',
    16: 'seagreen',
    17: 'yellowgreen',
    18: 'darksalmon',
    19: 'darkorange',
    20: 'coral',
}

# 创建类别映射表
voc = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

# 绘制GT bbox
plt.figure('Visualization')
plt.subplot(121)
plt.title('GT-bbox')
plt.imshow(test_img)
currentAxis = plt.gca()

for item in label:

    bbox_color = color_map[item['class']]

    pt1_x = (item['bbox'][0] - item['bbox'][2] / 2) * width
    pt1_y = (item['bbox'][1] - item['bbox'][3] / 2) * height

    pt1 = (pt1_x, pt1_y)  # 左上角点的坐标

    rect = patches.Rectangle(pt1, item['bbox'][2] * width, item['bbox'][3] * height,
                             linewidth=3, edgecolor=bbox_color, facecolor='none')

    plt.text(pt1_x, pt1_y, '{}:{:.2f}'.format(voc[item['class']], 1.0),
             fontdict={'size': 12, 'color': 'white', 'weight': 'bold'},
             bbox=dict(facecolor=bbox_color, alpha=1, edgecolor='none', pad=1.5))

    currentAxis.add_patch(rect)

# 绘制pred bbox
plt.subplot(122)
plt.title('pred-bbox')
plt.imshow(test_img)


if pred_loc.numel() != 0:  # 如果探测到物体才绘制bbox

    pred_conf, pred_class = torch.max(pred_conf, dim=1)

    # 使用NMS去除冗余的bbox
    pred_loc, pred_conf, pred_class = tools.nms(pred_loc, pred_conf, pred_class, nms_threshold)
    print('after nms {} objects left'.format(len(pred_class)))

    currentAxis = plt.gca()

    for index in range(len(pred_class)):
        bbox_color = color_map[pred_class[index]]

        pt1_x = (pred_loc[index][0] - pred_loc[index][2] / 2) * width
        pt1_y = (pred_loc[index][1] - pred_loc[index][3] / 2) * height

        pt1 = (pt1_x, pt1_y)

        rect = patches.Rectangle(pt1, pred_loc[index][2] * width, pred_loc[index][3] * height,
                                 linewidth=3, edgecolor=bbox_color, facecolor='none')
        plt.text(pt1_x, pt1_y, '{}:{:.2f}'.format(voc[pred_class[index]], pred_conf[index]),
                 fontdict={'size': 12, 'color': 'white', 'weight': 'bold'},
                 bbox=dict(facecolor=bbox_color, alpha=1, edgecolor='none', pad=1.5))

        currentAxis.add_patch(rect)

plt.show()