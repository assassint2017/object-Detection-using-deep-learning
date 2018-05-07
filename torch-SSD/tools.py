import math
import numpy as np
import torch

img_size = 300
fm_sizes = [38, 19, 10, 5, 3, 1]
steps = [s / img_size for s in (8, 16, 32, 64, 100, 300)]
sk = [s / img_size for s in (30, 60, 111, 162, 213, 264, 315)]

aspect_ratios = [  # 4 6 6 6 4 4
    [1, 2, 1/2],
    [1, 2, 1/2, 3, 1/3],
    [1, 2, 1/2, 3, 1/3],
    [1, 2, 1/2, 3, 1/3],
    [1, 2, 1/2],
    [1, 2, 1/2],
]

default_boxs = []


# 每一张图片有8732个default box，下面要计算出这些box的xywh相对坐标
for index, fm_size in enumerate(fm_sizes):
    for i in range(fm_size):
        for j in range(fm_size):
            x = (j + 0.5) * steps[index]
            y = (i + 0.5) * steps[index]

            for aspect_ratio in aspect_ratios[index]:

                w = sk[index] * math.sqrt(aspect_ratio)
                h = sk[index] / math.sqrt(aspect_ratio)

                default_boxs.append((x, y, w, h))

            # 最后一个box是扩大版的
            w = math.sqrt(sk[index] * sk[index + 1])
            h = math.sqrt(sk[index] * sk[index + 1])

            default_boxs.append((x, y, w, h))

# # 测试代码，可视化default box
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# img_size = 300
#
# img = np.ones((img_size, img_size, 3))  # 模拟图片
# plt.imshow(img)
#
# currentAxis = plt.gca()
#
# for item in default_boxs[:5775]:
#
#     pt1_x = (item[0] - item[2] / 2) * img_size
#     pt1_y = (item[1] - item[3] / 2) * img_size
#
#     pt1 = (pt1_x, pt1_y)  # 左上角点的坐标
#     width = item[2] * img_size
#     height = item[3] * img_size
#
#     rect = patches.Rectangle(pt1, width, height,
#                              linewidth=0.5, edgecolor='r', facecolor='none')
#
#     currentAxis.add_patch(rect)
#
# plt.show()


def iou(bbox1, bbox2):  # 计算两个bbox之间的IOU系数

    inter_width = (bbox1[2] / 2 + bbox2[2] / 2) - math.fabs(bbox1[0] - bbox2[0])

    inter_height = (bbox1[3] / 2 + bbox2[3] / 2) - math.fabs(bbox1[1] - bbox2[1])

    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    else:
        inter_area = inter_width * inter_height

        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]

        return inter_area / (bbox1_area + bbox2_area - inter_area)


def nms(pred_loc, pred_conf, pred_class, iou_threshold):
    """Non maximum suppression

    :param pred_loc: N,4
    :param pred_conf: N,1
    :param pred_class: N,1
    :param iou_threshold: IOU的阈值
    :return: 三个列表 loc conf classes

    Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    loc = []
    conf = []
    classes = []

    for class_index in range(1, 21):  # NMS要对所有物体类别都做一遍
        class_mask = pred_class == class_index

        if class_mask.type(torch.FloatTensor).sum() == 0:
            continue

        class_conf = pred_conf[class_mask]

        class_mask = class_mask.unsqueeze(1).expand_as(pred_loc)
        class_loc = pred_loc[class_mask].view(-1, 4)

        x = class_loc[:, 0]
        y = class_loc[:, 1]
        w = class_loc[:, 2]
        h = class_loc[:, 3]

        areas = w * h
        _, order = class_conf.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx = x[order[1:]]
            yy = y[order[1:]]
            ww = w[order[1:]]
            hh = h[order[1:]]

            width = ((ww / 2 + w[i] / 2) - torch.abs(xx - x[i])).clamp(min=0)
            height = ((hh / 2 + h[i] / 2) - torch.abs(yy - y[i])).clamp(min=0)
            inter = width * height

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]

        keep = torch.LongTensor(keep)

        class_conf = np.ndarray.tolist(class_conf[keep].cpu().detach().numpy())
        class_loc = np.ndarray.tolist(class_loc[keep].cpu().detach().numpy())

        for i in range(len(class_conf)):
            classes.append(class_index)

        loc += class_loc
        conf += class_conf

    return loc, conf, classes