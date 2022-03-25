import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.backbone import *
from libs.utils import *

class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv1d_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv1d_bn_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        # candidates
        self.sf = self.cfg.scale_factor

        candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))
        self.cand_c = to_tensor(candidates['c'])
        self.cand_mask = load_pickle(self.cfg.dir['pre3'] + 'candidate_mask_' + str(self.cfg.n_clusters))
        self.cand_iou = load_pickle(self.cfg.dir['pre3'] + 'candidate_iou_map_' + str(self.cfg.n_clusters))
        self.cand_iou_upper = load_pickle(self.cfg.dir['pre3'] + 'candidate_iou_upper_map_' + str(self.cfg.n_clusters))
        self.cand_iou = to_tensor(self.cand_iou)
        self.cand_iou_upper = to_tensor(self.cand_iou_upper)
        self.cand_area = dict()
        sf = cfg.scale_factor[0]
        self.cand_mask[sf], self.cand_area[sf] = self.get_lane_mask_area(self.cand_mask[sf])
        self.n_cand = self.cand_mask[sf].shape[2]

        self.c_feat = 512
        self.c_feat2 = 64

        self.c_sq = 32
        self.c_sq2 = 128

        # model
        self.encoder = resnet(layers=self.cfg.backbone, pretrained=True)
        backbone = self.cfg.backbone

        self.feat_squeeze1 = torch.nn.Sequential(
            conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
            conv_bn_relu(128, self.c_feat2, 3, padding=1),
        )
        self.feat_squeeze2 = torch.nn.Sequential(
            conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128, 3, padding=1),
            conv_bn_relu(128, self.c_feat2, 3, padding=1),
        )
        self.feat_squeeze3 = torch.nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, self.c_feat2, 3, padding=1),
        )
        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(self.c_feat2 * 3, 256, 3, padding=2, dilation=2),
            conv_bn_relu(256, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            torch.nn.Conv2d(128, self.c_sq, 1),
        )

        self.decoder = torch.nn.Sequential(
            conv_bn_relu(self.c_feat2 * 3, self.c_feat2 * 2, 3, padding=2, dilation=2),
            conv_bn_relu(self.c_feat2 * 2, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            torch.nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )

        # Cls & Reg
        self.classification1 = nn.Sequential(
            nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_sq),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_sq, 2, kernel_size=1, bias=False),
        )
        self.classification2 = nn.Sequential(
            nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_sq),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_sq, self.cfg.height_class.shape[0], kernel_size=1, bias=False),
        )

        self.regression1 = nn.Sequential(
            nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_sq),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False),
        )
        self.regression2 = nn.Sequential(
            nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_sq),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False),
        )
        self.regression3 = nn.Sequential(
            nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_sq),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False),
        )
        self.regression4 = nn.Sequential(
            nn.Conv1d(self.c_sq, self.c_sq, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_sq),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_sq, 1, kernel_size=1, bias=False),
        )

        self.w1 = nn.Sequential(
            nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_feat2 * 3),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False))
        self.w2 = nn.Sequential(
            nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.c_feat2 * 3),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.c_feat2 * 3, self.c_feat2 * 3, kernel_size=1, bias=False))

    def get_lane_mask_area(self, mask):
        n, h, w = mask.shape
        area = torch.zeros(n, dtype=torch.float32).cuda()
        for i in range(n):
            area[i] = mask[i].nonzero().shape[0]

        return mask.view(1, 1, n, h, w), area

    def lane_pooling(self, feat_map, idx, sf):
        b, c, h, w = feat_map.shape
        _, n = idx.shape

        mask = self.cand_mask[sf][:, :, idx].view(b, 1, n, h, w)
        area = self.cand_area[sf][idx].view(b, 1, n, 1, 1)

        line_feat = torch.sum(mask * feat_map.view(b, c, 1, h, w), dim=(3, 4), keepdim=True) / area
        return line_feat[:, :, :, 0, 0]

    def extract_lane_feat(self, feat_map, sf):
        b, c, h, w = feat_map.shape
        line_feat = torch.sum(self.cand_mask[sf][:, :, :] * feat_map.view(b, c, 1, h, w), dim=(3, 4)) / self.cand_area[sf][:].view(1, 1, -1)

        return line_feat

    def selection_and_removal(self, prob, batch_idx):
        idx_max = torch.sort(prob, descending=True, dim=1)[1][0, 0]
        cluster_idx = ((self.cand_iou[idx_max] >= self.thresd_nms_iou)).nonzero()[:, 0]

        if prob[0][idx_max] >= self.thresd_score:  # removal
            self.visit_mask[batch_idx, :, cluster_idx] = 0
            self.center_mask[batch_idx, :, idx_max] = 0

        return prob[0][idx_max], idx_max

    def forward_for_encoding(self, img):

        # Feature extraction
        feat1, feat2, feat3 = self.encoder(img)

        self.feat = dict()
        self.feat[self.sf[0]] = feat1
        self.feat[self.sf[1]] = feat2
        self.feat[self.sf[2]] = feat3

    def forward_for_decoding(self):
        out = self.decoder(self.x_concat)

        return {'seg_map': out}

    def forward_for_squeeze(self):

        # Feature squeeze and concat
        x1 = self.feat_squeeze1(self.feat[self.sf[0]])
        x2 = self.feat_squeeze2(self.feat[self.sf[1]])
        x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.feat_squeeze3(self.feat[self.sf[2]])
        x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode='bilinear')
        self.x_concat = torch.cat([x1, x2, x3], dim=1)
        self.sq_feat = self.feat_combine(self.x_concat)

    def forward_for_lane_feat_extraction(self):
        self.l_feat = self.extract_lane_feat(self.sq_feat, self.sf[0])

    def forward_for_lane_component_prediction(self):
        out1 = self.classification1(self.l_feat)
        out2 = self.classification2(self.l_feat)

        offset = list()
        offset.append(self.regression1(self.l_feat))
        offset.append(self.regression2(self.l_feat))
        offset.append(self.regression3(self.l_feat))
        offset.append(self.regression4(self.l_feat))

        offset = torch.cat(offset, dim=1)

        return {'prob': F.softmax(out1, dim=1)[:, 1:, :],
                'prob_logit': out1,
                'height_prob': F.softmax(out2, dim=1),
                'height_prob_logit': out2,
                'offset': offset.permute(0, 2, 1)}

    def forward_for_matching(self, idx):
        _, d = self.cand_c.shape
        batch_l_feat = self.lane_pooling(self.x_concat, idx, self.sf[0])
        out = self.correlation(batch_l_feat)
        for i in range(len(idx)):
            out[i].fill_diagonal_(0)

        return {'edge_map': out}

    def correlation(self, x):
        x1 = l2_normalization(self.w1(x))
        x2 = l2_normalization(self.w2(x))
        corr = torch.matmul(x1.permute(0, 2, 1), x2)

        return corr

def l2_normalization(x):
    ep = 1e-6
    out = x / (torch.norm(x, p=2, dim=1, keepdim=True) + ep)
    return out