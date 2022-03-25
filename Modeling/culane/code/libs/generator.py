import cv2
import torch

from libs.utils import *

class Label_Generator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))

        self.cand_px_coord = candidates['px_coord'].type(torch.float)
        self.cand_c = to_tensor(candidates['c'])
        self.cand_num = self.cand_c.shape[0]

        cand_mask = load_pickle(self.cfg.dir['pre3'] + 'candidate_mask_' + str(self.cfg.n_clusters))
        self.cand_mask = cand_mask[4]

        self.all_idx = set(np.arange(0, self.cand_num))

    def get_lane_mask(self, px_coord, sf=4, s=10):
        temp = np.zeros((self.cfg.height // sf, self.cfg.width // sf), dtype=np.float32)
        temp = np.ascontiguousarray(temp)

        x = px_coord / (self.cfg.width - 1) * (self.cfg.width // sf - 1)
        y = self.cfg.py_coord / (self.cfg.height - 1) * (self.cfg.height // sf - 1)

        xy_coord = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        xy_coord = np.int32(xy_coord).reshape((-1, 1, 2))

        lane_mask = cv2.polylines(temp, [xy_coord], False, 1, s)
        return to_tensor(lane_mask).unsqueeze(0)

    def measure_IoU(self, X1, X2):
        X = X1 + X2
        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)
        iou = X_inter / X_uni
        return iou

    def compute_iou_score(self):
        self.iou_score = torch.FloatTensor([]).cuda()
        for i in range(self.gt_num):
            lane_mask = self.get_lane_mask(to_np(self.org_px_coord[i]))
            h = lane_mask.shape[1]
            iou_score = self.measure_IoU(lane_mask[:, :h], self.cand_mask[:, :h])
            self.iou_score = torch.cat((self.iou_score, iou_score.view(1, -1)), dim=0)

    def get_data_list(self):
        self.check_pos = torch.zeros(self.cand_num, dtype=torch.long).cuda()
        self.check_adj = torch.zeros(self.cand_num, dtype=torch.long).cuda()

        self.idx_bin = torch.zeros(self.cand_num, max(1, self.gt_num), dtype=torch.float).cuda()

        self.pos_idx_list = list()
        self.adj_idx_list = list()
        self.iou_score_list = list()

        # compute iou
        self.compute_iou_score()
        if self.gt_num > 0:
            self.iou_max_score, self.iou_max_idx = torch.max(self.iou_score, dim=0)
        else:
            self.iou_max_score = torch.zeros(self.cand_num, dtype=torch.float).cuda()
            self.iou_max_score[:] = -1
            self.iou_max_idx = torch.LongTensor([]).cuda()

        # pos list
        self.pos_idx_list = ((self.cfg.thresd_label_pos_iou < self.iou_max_score)).nonzero()[:, 0]
        self.pos_lane_idx_list = self.iou_max_idx[self.pos_idx_list]

        self.check_pos[self.pos_idx_list] = 1
        self.idx_bin[self.pos_idx_list, self.pos_lane_idx_list] = 1

        # adj list
        self.adj_idx_list = ((self.cfg.thresd_label_adj_iou < self.iou_max_score) * (self.iou_max_score <= self.cfg.thresd_label_pos_iou)).nonzero()[:, 0]
        self.adj_lane_idx_list = self.iou_max_idx[self.adj_idx_list]

        self.check_adj[self.adj_idx_list] = 1
        self.idx_bin[self.adj_idx_list, self.adj_lane_idx_list] = 1

        # neg list
        self.neg_idx_list = ((self.check_pos + self.check_adj) == 0).nonzero()[:, 0]

    def get_pos_data(self):
        self.pos_idx = torch.LongTensor([]).cuda()
        self.pos_score = torch.FloatTensor([]).cuda()
        self.pos_cls = torch.FloatTensor([]).cuda()
        for i in range(self.gt_num):
            lane_idx_list = (self.pos_lane_idx_list == i).nonzero()[:, 0]
            idx_list = self.pos_idx_list[lane_idx_list]

            if idx_list.shape[0] == 0:
                lane_idx_list = torch.argmax(self.iou_score[i]).view(-1)
                idx_list = lane_idx_list
                self.idx_bin[idx_list, i] = 1

            iou_score_list = self.iou_max_score[lane_idx_list]
            rand_idx = torch.randperm(idx_list.shape[0])
            idx = idx_list[rand_idx[0]].view(-1)
            score = iou_score_list[rand_idx[0]]
            score = torch.FloatTensor([score]).cuda()
            cls = torch.FloatTensor([1]).cuda()

            self.pos_idx = torch.cat((self.pos_idx, idx), dim=0)
            self.pos_score = torch.cat((self.pos_score, score), dim=0)
            self.pos_cls = torch.cat((self.pos_cls, cls), dim=0)

    def get_node_data(self):
        self.node_score = self.iou_max_score

    def get_not_pos_data(self):
        nms_not_pos_idx = set(to_np(self.out_idx)) - set(to_np(self.pos_idx_list))

        # add rest
        rest_num = max(0, self.cfg.max_iter_train - self.gt_num - len(nms_not_pos_idx))
        if rest_num != 0:
            rest_idx_list = self.all_idx - set(to_np(self.pos_idx_list)) - nms_not_pos_idx  # among adj + neg except for nms
            rand_idx = torch.randperm(len(rest_idx_list))
            temp_idx = set(np.array(list(rest_idx_list))[rand_idx][:rest_num])
            nms_not_pos_idx = nms_not_pos_idx.union(temp_idx)

        nms_not_pos_idx = torch.LongTensor(list(nms_not_pos_idx)).cuda()

        rand_idx = torch.randperm(nms_not_pos_idx.shape[0])
        self.not_pos_idx = nms_not_pos_idx[rand_idx][:self.cfg.max_iter_train - self.gt_num]
        self.not_pos_cls = torch.zeros(self.cfg.max_iter_train - self.gt_num, dtype=torch.float).cuda()

    def get_edge_data(self):

        rand_idx = torch.randperm(self.cfg.max_iter_train)

        self.node_idx = torch.cat((self.pos_idx, self.not_pos_idx), dim=0)
        self.edge_score = self.node_score[self.node_idx]
        self.hard_cls = torch.cat((self.pos_cls, self.not_pos_cls), dim=0)
        self.soft_cls = torch.cat((self.pos_cls, self.check_adj[self.not_pos_idx].type(torch.float)), dim=0)

        self.node_idx = self.node_idx[rand_idx]
        self.edge_score = self.edge_score[rand_idx].view(-1, 1)
        self.hard_cls = self.hard_cls[rand_idx].view(-1, 1)
        self.soft_cls = self.soft_cls[rand_idx].view(-1, 1)
        self.node_bin = self.idx_bin[self.node_idx]


        self.adj_pair_mask = torch.matmul(self.node_bin, self.node_bin.permute(1, 0))
        self.adj_pair_mask.fill_diagonal_(0)
        self.hard_cls_mask = torch.matmul(self.hard_cls, self.hard_cls.permute(1, 0))
        self.soft_cls_mask = torch.matmul(self.soft_cls, self.soft_cls.permute(1, 0))

        self.avg_score = (self.edge_score + self.edge_score.permute(1, 0)) / 2
        self.avg_score -= self.avg_score * self.adj_pair_mask
        self.avg_score -= self.avg_score * (~self.soft_cls_mask.type(torch.bool)).type(torch.float)

        self.edge_map = self.avg_score
        self.edge_map.fill_diagonal_(0)

    def get_label_distribution(self, data):
        edge_count = np.zeros(11, dtype=np.int32)
        tmp = torch.round(data['gt_edge_map'] * 10)
        for i in range(11):
            edge_count[i] = torch.sum(tmp == i)
        return {'edge_count': edge_count}

    def run(self, batch, data):
        b = len(data['center_idx'])
        out = dict(
            node_idx=torch.LongTensor([]).cuda(),
            gt_edge_map=torch.FloatTensor([]).cuda()
        )

        for i in range(b):
            self.out_idx = data['center_idx'][i]
            self.exist = batch['exist'][i]
            self.gt_num = int(torch.sum(self.exist))

            self.gt_c = batch['gt_c'][i][:self.gt_num]
            self.org_px_coord = batch['org_px_coord'][i][:self.gt_num]

            self.get_data_list()
            self.get_pos_data()
            self.get_not_pos_data()
            self.get_node_data()
            self.get_edge_data()

            n = self.edge_map.shape[0]
            out['gt_edge_map'] = torch.cat((out['gt_edge_map'], self.edge_map.view(1, n, n)), dim=0)
            out['node_idx'] = torch.cat((out['node_idx'], self.node_idx.view(1, n)), dim=0)

        out.update(self.get_label_distribution(out))

        return out
