import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


num_class = 21  # 数据集物体的类别数目

fm_sizes = [38, 19, 10, 5, 3, 1]
num_anchor = [4, 6, 6, 6, 4, 4]


class L2norm(nn.Module):
    def __init__(self, scale=20):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(512))
        self.eps = 1e-10

        # 权重值初始化为20
        nn.init.constant(self.weight, self.scale)

    def forward(self, inputs):

        l2norm = torch.sqrt(torch.sum(torch.pow(inputs, 2), dim=1, keepdim=True)) + self.eps

        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * (inputs / l2norm)


class SSD300(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(3, 1, 1),

            nn.Conv2d(512, 1024, 3, dilation=6, padding=6),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, 1),
            nn.ReLU()
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU()
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )

        self.stage6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )

        # l2归一化层
        self.l2norm = L2norm()

        # 得到类别信息和偏移信息的卷积核
        self.clf1_conf = nn.Conv2d(512, 4 * num_class, 3, padding=1)
        self.clf1_loc = nn.Conv2d(512, 4 * 4, 3, padding=1)

        self.clf2_conf = nn.Conv2d(1024, 6 * num_class, 3, padding=1)
        self.clf2_loc = nn.Conv2d(1024, 6 * 4, 3, padding=1)

        self.clf3_conf = nn.Conv2d(512, 6 * num_class, 3, padding=1)
        self.clf3_loc = nn.Conv2d(512, 6 * 4, 3, padding=1)

        self.clf4_conf = nn.Conv2d(256, 6 * num_class, 3, padding=1)
        self.clf4_loc = nn.Conv2d(256, 6 * 4, 3, padding=1)

        self.clf5_conf = nn.Conv2d(256, 4 * num_class, 3, padding=1)
        self.clf5_loc = nn.Conv2d(256, 4 * 4, 3, padding=1)

        self.clf6_conf = nn.Conv2d(256, 4 * num_class, 3, padding=1)
        self.clf6_loc = nn.Conv2d(256, 4 * 4, 3, padding=1)

    def forward(self, inputs):

        locs = []
        confs = []

        temp = self.stage1(inputs)

        norm = self.l2norm(temp)

        confs.append(self.clf1_conf(norm))
        locs.append(self.clf1_loc(norm))

        temp = self.stage2(temp)

        confs.append(self.clf2_conf(temp))
        locs.append(self.clf2_loc(temp))

        temp = self.stage3(temp)

        confs.append(self.clf3_conf(temp))
        locs.append(self.clf3_loc(temp))

        temp = self.stage4(temp)

        confs.append(self.clf4_conf(temp))
        locs.append(self.clf4_loc(temp))

        temp = self.stage5(temp)

        confs.append(self.clf5_conf(temp))
        locs.append(self.clf5_loc(temp))

        temp = self.stage6(temp)

        confs.append(self.clf6_conf(temp))
        locs.append(self.clf6_loc(temp))

        # 数据维度变换
        for index in range(len(locs)):
            locs[index] = locs[index].permute(0, 2, 3, 1).contiguous()
            locs[index] = locs[index].view(-1, fm_sizes[index] ** 2 * num_anchor[index], 4)

        for index in range(len(confs)):
            confs[index] = confs[index].permute(0, 2, 3, 1).contiguous()
            confs[index] = confs[index].view(-1, fm_sizes[index] ** 2 * num_anchor[index], num_class)

        pred_loc = torch.cat(locs, dim=1)
        pred_conf = torch.cat(confs, dim=1)

        return pred_conf, pred_loc


def init(module):  # 参数初始化
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform(module.weight.data)
        nn.init.constant(module.bias.data, 0)


ssd = SSD300()
ssd.apply(init)

# 把预训练的网络参数加载到新创建的网络
vgg16 = torchvision.models.vgg16(pretrained=True)

conv_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]  # 卷积层的编号

for index in conv_layers:
    ssd.stage1[index].weight.data = vgg16.features[index].weight.data
    ssd.stage1[index].bias.data = vgg16.features[index].bias.data

ssd.stage2[1].weight.data = vgg16.features[24].weight.data
ssd.stage2[1].bias.data = vgg16.features[24].bias.data

ssd.stage2[3].weight.data = vgg16.features[26].weight.data
ssd.stage2[3].bias.data = vgg16.features[26].bias.data

ssd.stage2[5].weight.data = vgg16.features[28].weight.data
ssd.stage2[5].bias.data = vgg16.features[28].bias.data

# # 测试代码：查看网络输入数据维度
# data = torch.ones((1, 3, 300, 300)).cuda()
# ssd = ssd.cuda()
# pred_conf, pred_loc = ssd(data)
# print(pred_conf.size(), pred_loc.size())