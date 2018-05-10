from time import time
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from ssd300 import ssd

from dataset import train_ds

from LCloss import LCloss


# 定义超参数
Epoch = 174
learning_rate_base = 1e-3
weight_decay = 5e-4
momentum = 0.9

on_server = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '0,1'  # 设置在服务器端运行网络的GPU设备号码
batch_size = 16 if on_server is False else 32
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True

cudnn.benchmark = True


# 定义网络
net = torch.nn.DataParallel(ssd)
net = net.cuda()

# 定义训练数据加载
train_dl = DataLoader(train_ds, batch_size, True,
                      num_workers=num_workers, pin_memory=pin_memory)

# 定义优化器
opt = torch.optim.SGD(ssd.parameters(), lr=learning_rate_base,
                      momentum=momentum, weight_decay=weight_decay)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [115, 144])

# 实例化一个loss
lcloss = LCloss()

# 训练网络
start = time()

for epoch in range(Epoch):

    lr_decay.step()

    for step, (img, tar_conf, tar_loc), in enumerate(train_dl):

        if on_server is True:
            img = Variable(img)
            tar_conf = Variable(tar_conf)
            tar_loc = Variable(tar_loc)

        img = img.cuda()
        tar_conf = tar_conf.cuda()
        tar_loc = tar_loc.cuda()

        pred_conf, pred_loc = net(img)
        loss = lcloss(pred_conf, pred_loc, tar_conf, tar_loc)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # 每20步打印一次信息
        if step % 20 is 0:

            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.data[0], (time() - start) / 60))

    # 每一个epoch保存一次模型参数
    torch.save(net.state_dict(), './module/net{}-{:.3f}.pth'.format(epoch, loss.data[0]))