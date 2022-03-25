import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utils import *

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll1 = nn.NLLLoss(reduce=True)
        self.nll2 = nn.NLLLoss(reduce=False)

    def forward(self, logits, labels, reduce=True):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score

        if reduce == True:
            loss = self.nll1(log_score, labels)
        else:
            loss = self.nll2(log_score, labels)
        return loss

class Loss_Function(nn.Module):
    def __init__(self, cfg):
        super(Loss_Function, self).__init__()
        self.cfg = cfg

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_nce = nn.CrossEntropyLoss()
        self.loss_score = nn.MSELoss()
        self.loss_focal = SoftmaxFocalLoss(gamma=2)

        self.weight = 1

        offset = load_pickle(cfg.dir['pre4'] + 'offset_distribution_{}_{}_{}'.format(cfg.thresd_iou_for_reg, cfg.thresd_min_offset, cfg.thresd_max_offset))
        self.c_weight = 1 / (np.abs(offset['mean']) / np.abs(offset['mean']).max())

        candidates = load_pickle(cfg.dir['pre3'] + 'lane_candidates_' + str(cfg.n_clusters))
        self.cand_c = to_tensor(candidates['c'])

    def forward(self, out, gt):
        loss_dict = dict()

        l_prob = self.loss_focal(out['prob_logit'], gt['prob'])

        l_offset_tot = torch.FloatTensor([0.0]).cuda()
        for i in range(self.cfg.top_m):
            l_offset = self.balanced_MSE_loss(out['offset'][:, :, i:i + 1], gt['offset'][:, :, i:i + 1], gt['is_pos_reg']) * 0.1 * self.c_weight[i]
            loss_dict['offset' + str(i+1)] = l_offset
            l_offset_tot += l_offset

        l_edge = self.loss_mse(out['edge_map'], out['gt_edge_map'])
        l_seg = self.loss_bce(out['seg_map'][:, 0], gt['seg_label'])

        if torch.sum(gt['exist_check']) > 0:
            val = self.loss_focal(out['height_prob_logit'][gt['exist_check'] == 1], gt['height_prob'][gt['exist_check'] == 1], reduce=False) \
                  * gt['prob'][gt['exist_check'] == 1].type(torch.float)
            l_prob_h = torch.mean(torch.sum(val, dim=1) / torch.sum(gt['prob'][gt['exist_check'] == 1].type(torch.float), dim=1), dim=0) * 0.01
        else:
            l_prob_h = torch.FloatTensor([0.0]).cuda()

        loss_dict['sum'] = l_prob + l_offset_tot + l_seg + l_edge + l_prob_h
        loss_dict['prob'] = l_prob
        loss_dict['prob_h'] = l_prob_h
        loss_dict['edge'] = l_edge
        loss_dict['seg'] = l_seg

        return loss_dict

    def balanced_MSE_loss(self, out, gt, gt_check):
        ep = 1e-6
        neg_mask = (gt_check == 0).unsqueeze(2)
        pos_mask = (gt_check != 0).unsqueeze(2)
        neg_num = torch.sum(neg_mask, dim=(1, 2)) + ep
        pos_num = torch.sum(pos_mask, dim=(1, 2)) + ep
        pos_loss = torch.mean(torch.sum(F.mse_loss(out * pos_mask, gt * pos_mask, reduce=False), dim=(1, 2)) / pos_num)
        neg_loss = torch.mean(torch.sum(F.mse_loss(out * neg_mask, gt * neg_mask, reduce=False), dim=(1, 2)) / neg_num)

        return pos_loss + neg_loss
