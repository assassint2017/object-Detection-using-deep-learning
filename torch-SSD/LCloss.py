import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class LCloss(nn.Module):
    """

    LCloss由两部分组成，第一个是loc定位损失，另一部分是conf分类损失
    """
    def __init__(self, ):
        super().__init__()

    def forward(self, pred_conf, pred_loc, tar_conf, tar_loc):
        """

        :param pred_conf: 预测得到的分类概率向量---B, 8732, 21
        :param pred_loc: 预测得到的偏移向量---B, 8732, 4
        :param tar_conf: 目标的分类概率向量---B, 8732,
        :param tar_loc: 目标的偏移向量---B, 8732, 4
        """

        pos = tar_conf > 0  # B, 8732,

        neg = tar_conf == 0  # B, 8732,

        num_match = torch.sum(pos.type(torch.FloatTensor)).cuda()

        if num_match == 0:
            # 如果没有匹配的loss就为0
            return Variable(torch.FloatTensor([0]).cuda())

        # 求取定位损失
        pos_mask = pos.unsqueeze(2).expand_as(pred_loc)

        pred_loc = pred_loc[pos_mask].view(-1, 4)

        tar_loc = tar_loc[pos_mask].view(-1, 4)

        loc_loss = F.smooth_l1_loss(pred_loc, tar_loc, size_average=False)
        loc_loss /= num_match

        # 求取正样本bbox的分类损失
        pos_mask = pos.unsqueeze(2).expand_as(pred_conf)

        pos_pred_conf = pred_conf[pos_mask].view(-1, 21)

        pos_tar_conf = tar_conf[pos]

        pos_conf_loss = F.cross_entropy(pos_pred_conf, pos_tar_conf)

        # 利用Hard negative mining求取负样本bbox的分类损失，负样本的数量是正样本数量的三倍
        neg_mask = neg.unsqueeze(2).expand_as(pred_conf)
        neg_pred_conf = pred_conf[neg_mask].view(-1, 21)

        neg_mask = torch.cat([torch.ones((int(num_match) * 3, 21)).type(torch.ByteTensor),
                              torch.zeros((neg_pred_conf.size(0) - int(num_match) * 3, 21)).type(torch.ByteTensor)],
                             dim=0).cuda()

        neg_pred_conf = neg_pred_conf.sort(0, descending=False)[0][neg_mask].view(-1, 21)

        # 背景是第0类，所有标签是0
        neg_tar_conf = Variable(torch.zeros(neg_pred_conf.size(0)).type(torch.LongTensor).cuda())

        neg_cong_loss = F.cross_entropy(neg_pred_conf, neg_tar_conf)

        # 总的loss由三部分共同构成
        loss = pos_conf_loss + neg_cong_loss + loc_loss

        return loss
