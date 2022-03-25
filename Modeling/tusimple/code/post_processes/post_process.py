import os
import cv2
import torch
import math

import numpy as np

from libs.utils import *


class Post_Processing(object):

    def __init__(self, cfg):
        self.cfg = cfg

        self.encode_bitlist()
        self.decode_bitlist()
        self.encode_edge_idx()
        self.match_bitlist_with_edge()

        # candidates
        candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))
        self.cand_c = to_tensor(candidates['c'])
        self.U = load_pickle(self.cfg.dir['pre3'] + 'U')[:, :self.cfg.top_m]

        # others
        self.tau = 0.5
        self.MIN_VAL = -1000
        self.upper_tri = torch.triu(torch.ones(self.cfg.max_iter, self.cfg.max_iter), diagonal=1)

    def encode_bitlist(self):
        self.num_node = self.cfg.max_iter
        num_clique = 2 ** self.num_node

        bitlist = torch.LongTensor([]).cuda()
        for i in range(1, num_clique):

            k = i
            bit = torch.zeros((1, self.num_node), dtype=torch.int64).cuda()
            for j in range(self.num_node):
                rest = k % 2
                k //= 2
                bit[0, j] = rest
                if k == 0:
                    break
            if torch.sum(bit) == 1:  # case of node
                continue

            if self.cfg.constrain_max_lane_num == True:
                if torch.sum(bit) >= self.cfg.max_lane_num + 1:  # case of node
                    continue
            bitlist = torch.cat((bitlist, bit))

        self.bitlist = bitlist
        self.num_clique = self.bitlist.shape[0]

    def decode_bitlist(self):
        check = torch.zeros((2 ** self.num_node), dtype=torch.int64).cuda()
        for i in range(self.num_clique):

            bit = self.bitlist[i]
            k = 1
            m = 0
            for j in range(self.num_node):
                m += (k * bit[j])
                k *= 2
            check[m] = 1

        idx = (check == 0).nonzero()
        if idx.shape[0] == 0:
            print('Successfully encoded')
        else:
            print("error code : {}".format(idx))

    def encode_edge_idx(self):
        edge_idx = torch.zeros((self.num_node, self.num_node), dtype=torch.int64).cuda()
        k = 0
        for i in range(self.num_node):
            for j in range(i + 1, self.num_node):
                edge_idx[i, j] = k
                k += 1

        self.edge_idx = edge_idx
        self.edge_max_num = torch.max(edge_idx)

    def match_bitlist_with_edge(self):

        clique_idxlist = torch.LongTensor([]).cuda()
        for i in range(self.num_clique):

            bit = self.bitlist[i]
            nodelist = bit.nonzero()
            num_node = nodelist.shape[0]
            idx_check = torch.zeros((1, self.edge_max_num + 1), dtype=torch.int64).cuda()
            for j in range(num_node):
                for k in range(j + 1, num_node):
                    idx_check[0, self.edge_idx[nodelist[j, 0], nodelist[k, 0]]] = 1

            clique_idxlist = torch.cat((clique_idxlist, idx_check))

        self.clique_idxlist = clique_idxlist
        self.clique_idxnum = torch.sum(clique_idxlist, dim=1)

    def run_for_mwcs(self):
        clique_energy = torch.sum((self.edge_score * self.clique_idxlist), dim=1)

        idx_max = torch.argmax(clique_energy)
        if clique_energy[idx_max] > self.tau:
            mul_idx = self.bitlist[idx_max].nonzero()[:, 0]
        else:
            if torch.max(self.prob) > 0.5:
                mul_idx = torch.LongTensor([torch.argmax(self.prob)]).cuda()
            else:
                mul_idx = torch.LongTensor([]).cuda()
        return {'match_idx': [self.node_idx[mul_idx]],
                'match_nms_idx': [mul_idx]}

    def run_for_c_to_x_coord_conversion(self, idx, offset=None, key=None):
        if idx.shape[0] != 0:
            cand_c = self.cand_c[idx]
            if offset is not None:
                reg_c = cand_c + offset
            else:
                reg_c = cand_c
            px_coord = torch.matmul(self.U, reg_c.permute(1, 0)).permute(1, 0) * (self.cfg.width - 1)
            px_coord = to_np2(px_coord)
        else:
            px_coord = np.float32([])

        return {key[0]: px_coord}

    def run_for_cls_height(self, data, key):
        out = dict()

        idx = data[key[1]][0]

        if len(idx) != 0:
            height_idx = torch.argmax(data['height_prob'][0][:, idx], dim=0)

            height_pred_idx = to_np(height_idx)
            height_node_idx = self.cfg.height_node_idx[to_np(height_idx)]

            out[key[0] + '_pred_idx'] = height_pred_idx
            out[key[0] + '_idx'] = height_node_idx
        else:
            out[key[0] + '_pred_idx'] = np.int64([])
            out[key[0] + '_idx'] = np.int64([])

        return out

    def run_for_vp(self, out, key):
        vp_idx = []
        px_coord = out[key[0]]
        if px_coord.shape[0] > 1:
            vp_idx_tmp = np.argmin(np.mean(np.abs(np.mean(px_coord, axis=0, keepdims=True) - np.array(px_coord)), axis=0))
            vp_idx.append(vp_idx_tmp)
        else:
            vp_idx.append(0)

        return {key[0] + '_vp_idx': np.int64(vp_idx)}

    def run(self, out):
        out.update(self.run_for_mwcs())
        out.update(self.run_for_c_to_x_coord_conversion(idx=out['center_idx'][0], key=['nms']))
        out.update(self.run_for_c_to_x_coord_conversion(idx=out['center_idx'][0], offset=out['offset'][0][out['center_idx'][0]], key=['nms_reg']))
        out.update(self.run_for_c_to_x_coord_conversion(idx=out['match_idx'][0], key=['mwcs']))
        out.update(self.run_for_c_to_x_coord_conversion(idx=out['match_idx'][0], offset=out['offset'][0][out['match_idx'][0]], key=['mwcs_reg']))
        out.update(self.run_for_cls_height(out, key=['nms_height', 'center_idx']))
        out.update(self.run_for_cls_height(out, key=['mwcs_height', 'match_idx']))
        out.update(self.run_for_vp(out, key=['mwcs_reg']))
        return out

    def update(self, batch, out, mode):
        self.mode = mode
        self.img_name = batch['img_name'][0]

        self.edge_score = (out['edge_map'][0] + out['edge_map'][0].transpose(1, 0)) / 2
        self.edge_score = self.edge_score[self.upper_tri == 1]
        idx = (self.edge_score <= self.tau)
        self.edge_score[idx] = self.MIN_VAL
        self.edge_score = self.edge_score.view(1, -1)
        self.node_idx = out['center_idx'][0]
        self.prob = out['prob'][0, 0, self.node_idx]
