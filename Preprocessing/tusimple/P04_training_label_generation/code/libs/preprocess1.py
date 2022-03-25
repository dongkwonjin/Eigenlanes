import cv2
import math

import torch
import torch.nn.functional as F

from libs.utils import *

from sklearn.cluster import KMeans

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualize = dict_DB['visualize']

        self.min_offset = 99999
        self.max_offset = -99999

    def load_default_data(self):
        self.datalist = load_pickle(self.cfg.dir['pre2'] + 'datalist')

        candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))
        self.cand_px_coord = candidates['px_coord'].type(torch.float)
        self.cand_c = to_tensor(candidates['c'])

        self.cand_mask = self.get_cand_lane_mask(to_np(self.cand_px_coord))
        self.cand_area = dict()
        for sf in self.cfg.scale_factor:
            self.cand_area[sf] = torch.sum(self.cand_mask[sf], dim=(1, 2))

        self.U = load_pickle(self.cfg.dir['pre2'] + 'U')
        self.py_coord = to_tensor(self.cfg.py_coord)

    def load_preprocessed_data(self):
        self.org_lane = load_pickle(self.cfg.dir['pre1'] + self.img_name)[self.flip_idx]
        self.coefficient_vector = load_pickle(self.cfg.dir['pre2'] + self.img_name)[self.flip_idx]

    def get_cand_lane_mask(self, px_coord):
        out_mask = torch.FloatTensor([]).cuda()
        for i in range(px_coord.shape[0]):
            mask = self.get_lane_mask(px_coord[i])
            out_mask = torch.cat((out_mask, mask), dim=0)
        return {self.cfg.scale_factor[0]: out_mask}

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

    def get_label(self):
        out = {'iou': [], 'iou_upper': [], 'theta': [], 'error': [], 'height': [],
               'exist': [], 'org_px_coord': [], 'org_c': []}

        k = 0

        for i in range(len(self.org_lane['x_coord'])):
            px_coord = to_tensor(self.org_lane['x_coord'][i])[self.cfg.sample_idx]
            c = self.coefficient_vector['c'][:, k]

            lane_mask = self.get_lane_mask(to_np(px_coord))
            h = self.cfg.height // 4
            h2 = int((self.cfg.py_coord[0] + self.cfg.max_y_coord / 3) / 4)
            iou = self.measure_IoU(lane_mask[:, :h], self.cand_mask[4][:, :h])
            iou_upper = self.measure_IoU(lane_mask[:, :h2], self.cand_mask[4][:, :h2])
            nan_idx = torch.isnan(iou_upper).nonzero()[:, 0]
            iou_upper[nan_idx] = 1
            if nan_idx.shape[0] > 0:
                print('solve nan error')

            if np.sum(np.isnan(to_np(iou_upper))) > 0:
                print('occur nan error')

            error = c.view(1, -1) - self.cand_c
            height = self.org_lane['height'][i]

            out['error'].append(to_np(error))
            out['iou'].append(to_np(iou))
            out['iou_upper'].append(to_np(iou_upper))
            out['org_px_coord'].append(to_np(px_coord))
            out['org_c'].append(to_np(c))
            out['height'].append(height)
            out['exist'].append(1)

            k += 1

        return out

    def run(self):
        print('start')

        datalist = []
        self.load_default_data()
        for i in range(len(self.datalist)):
            self.img_name = self.datalist[i]
            out_f = list()
            for j in range(0, 2):  # 1: horizontal flip
                if j == 1 and self.cfg.data_flip == False:
                    continue
                if j == 1 and self.cfg.datalist == 'test':
                    break
                self.flip_idx = j

                self.load_preprocessed_data()
                out_f.append(self.get_label())

            if self.cfg.save_pickle == True:
                save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name=self.img_name, data=out_f)
                datalist.append(self.img_name)

            print("image {} ===> {} clear".format(i, self.img_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist_all', data=datalist)

