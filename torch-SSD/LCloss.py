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

        num_match = torch.sum(pos.type(torch.FloatTensor)).cuda()   # 一个batch里总的正样板的数量

        if num_match == 0:
            # 如果没有匹配的loss就为0
            return Variable(torch.FloatTensor([0]).cuda())

        # 求取定位损失
        pos_mask = pos.unsqueeze(2).expand_as(pred_loc)

        pred_loc = pred_loc[pos_mask].view(-1, 4)

        tar_loc = tar_loc[pos_mask].view(-1, 4)

        loc_loss = F.smooth_l1_loss(pred_loc, tar_loc, size_average=False)

        # 求取样本分类损失
        conf_loss = F.cross_entropy(input=pred_conf.view(-1, 21),
                                    target=tar_conf.view(-1),
                                    reduce=False)  # B * 8732

        conf_loss = conf_loss.view(-1, 8732)  # B, 8732

        conf_loss[pos] = 0

        _, idx = conf_loss.sort(dim=1, descending=True)
        _, rank = idx.sort(dim=1)

        num_match_box = pos.type(torch.LongTensor).sum(dim=1).cuda()
        num_neg_box = num_match_box * 3

        neg = rank < num_neg_box.unsqueeze(dim=1).expand_as(rank)  # B, 8732

        conf_loss = F.cross_entropy(input=pred_conf[(neg + pos).unsqueeze(2).expand_as(pred_conf)].view(-1, 21),
                                    target=tar_conf[(neg + pos)].view(-1),
                                    size_average=False)

        # loss 求取平均
        conf_loss /= num_match
        loc_loss /= num_match

        # 总的loss由两部分共同组成
        return conf_loss + loc_loss

