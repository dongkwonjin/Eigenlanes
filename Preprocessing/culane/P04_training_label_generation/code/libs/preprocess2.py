import cv2
import math

import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualize = dict_DB['visualize']

        self.min_offset = 99999
        self.max_offset = -99999

    def load_default_data(self):
        self.datalist = load_pickle(self.cfg.dir['pre2'] + 'datalist')

        candidates = load_pickle(self.cfg.dir['pre3'] + 'lane_candidates_' + str(self.cfg.n_clusters))
        self.cand_px_coord = candidates['px_coord'].type(torch.float)
        self.cand_c = to_tensor(candidates['c'])

        self.cand_mask = load_pickle(self.cfg.dir['pre3'] + 'candidate_mask_' + str(self.cfg.n_clusters))
        self.cand_area = dict()
        for sf in self.cfg.scale_factor:
            self.cand_area[sf] = torch.sum(self.cand_mask[sf], dim=(1, 2))
            n, h, w = self.cand_mask[sf].shape
            self.cand_mask[sf] = self.cand_mask[sf].view(1, 1, n, h, w)

        self.U = load_pickle(self.cfg.dir['pre2'] + 'U')
        self.py_coord = to_tensor(self.cfg.py_coord)

    def load_preprocessed_data(self):
        self.org_lane = load_pickle(self.cfg.dir['pre1'] + self.img_name)[self.flip_idx]
        self.coefficient_vector = load_pickle(self.cfg.dir['pre2'] + self.img_name)[self.flip_idx]
        self.lane_dist = load_pickle(self.cfg.dir['out'] + 'pickle/' + self.img_name)[self.flip_idx]

    def run(self):
        print('start')

        datalist = []
        datalist_error = []
        self.load_default_data()

        for i, batch in enumerate(self.dataloader):
            self.img = batch['img'][0].cuda()
            self.label = batch['label'][0].cuda()
            self.img_name = batch['img_name'][0]

            self.is_error_case = 0

            for j in range(0, 2):  # 1: horizontal flip

                if j == 1 and self.cfg.data_flip == False:
                    continue
                if j == 1 and self.cfg.datalist == 'test':
                    break
                self.flip_idx = j

                if j == 1:
                    self.img = self.img.flip(2)
                    self.label = self.label.flip(1)

                self.load_preprocessed_data()
                self.visualize.update_datalist(self.img, self.img_name, self.label)

                temp = {'cand_px_coord': [], 'org_px_coord': [], 'cand_c': [], 'offset': []}

                for k in range(len(self.org_lane['x_coord'])):
                    if len(self.org_lane['x_coord'][k]) == 0:
                        temp['cand_px_coord'].append([])
                        temp['org_px_coord'].append([])
                        temp['cand_c'].append([])
                        temp['offset'].append([])
                        continue

                    iou = to_tensor(self.lane_dist['iou'][k])
                    error = to_tensor(self.lane_dist['error'][k])
                    px_coord = to_tensor(self.org_lane['x_coord'][k])[self.cfg.sample_idx]

                    idx_max = torch.argsort(iou, descending=True)[0]

                    idx = ((iou > self.cfg.thresd_iou)).nonzero()[:, 0]
                    if idx.shape[0] == 0:
                        idx = torch.LongTensor([idx_max]).cuda()

                    iou_max = iou[idx_max]
                    if iou_max < self.cfg.thresd_iou_error:
                        self.is_error_case = 1

                    offset = error[idx]

                    check1 = (torch.min(offset, dim=1)[0] >= self.cfg.thresd_min_offset)
                    check2 = (torch.max(offset, dim=1)[0] <= self.cfg.thresd_max_offset)
                    offset_check = check1 * check2

                    if torch.sum(offset_check) == 0:
                        self.is_error_case = 2
                    else:
                        idx = idx[offset_check == True]
                        offset = error[idx]

                    cand_px_coord = self.cand_px_coord[idx]
                    cand_c = self.cand_c[idx]

                    temp['cand_px_coord'].append(cand_px_coord)
                    temp['org_px_coord'].append(px_coord)
                    temp['cand_c'].append(cand_c)
                    temp['offset'].append(offset)

                    if self.is_error_case == 0:
                        if torch.max(offset) > self.max_offset:
                            self.max_offset = torch.max(offset)
                        if torch.min(offset) < self.min_offset:
                            self.min_offset = torch.min(offset)

                if self.is_error_case != 0 or self.cfg.display_all == True:
                    self.visualize.draw_lanes_for_datalist(temp, self.U, self.org_lane, self.py_coord)

            if self.cfg.display_all == True or self.is_error_case != 0:
                self.visualize.save_datalist(self.is_error_case, i)

            if self.is_error_case == 0:
                datalist.append(self.img_name)
            else:
                datalist_error.append(self.img_name)

            print("image {} ===> {} clear min-max offset {} {}".format(i, self.img_name, self.min_offset, self.max_offset))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist', data=datalist)
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist_error', data=datalist_error)

        eigenlane_distribution(self.cfg)

def eigenlane_distribution(cfg):
    datalist = load_pickle(cfg.dir['out'] + 'pickle/datalist')

    t1 = cfg.thresd_iou

    to1 = cfg.thresd_min_offset
    to2 = cfg.thresd_max_offset

    X1 = np.zeros((cfg.top_m), dtype=np.float32)
    X2 = np.zeros((cfg.top_m), dtype=np.float32)
    num = 0
    for i in range(len(datalist)):
        img_name = datalist[i]
        data = load_pickle(cfg.dir['out'] + 'pickle/' + img_name)
        for j in range(0, 2):  # 1: horizontal flip
            if j == 1 and cfg.data_flip == False:
                continue
            if j == 1 and cfg.datalist == 'test':
                break

            iou = data[j]['iou']
            error = data[j]['error']
            for k in range(len(data[j]['iou'])):
                if len(iou[k]) == 0:
                    continue

                idx_max = torch.argsort(to_tensor(iou[k]), descending=True)[0]

                idx = ((iou[k] > t1)).nonzero()[0]
                if idx.shape[0] == 0:
                    idx = torch.LongTensor([idx_max]).cuda()

                offset = error[k][idx]
                if len(offset.shape) == 1:
                    offset = offset.reshape(1, -1)

                check1 = (np.min(offset, axis=1) >= to1)
                check2 = (np.max(offset, axis=1) <= to2)
                offset_check = check1 * check2

                if np.sum(offset_check) == 0:
                    idx = torch.LongTensor([idx_max]).cuda()
                else:
                    idx = idx[offset_check == True]
                offset = error[k][idx]

                X1 += np.sum(offset, axis=0)
                X2 += np.sum(np.square(offset), axis=0)
                num += offset.shape[0]

        print('%d done' % i)

    mean_X1 = X1 / num
    mean_X2 = X2 / num

    mean = np.copy(mean_X1)
    var = mean_X2 - mean_X1 * mean_X1
    data = {}
    data['mean'] = mean
    data['var'] = var
    print(mean, var)
    if cfg.save_pickle == True:
        save_pickle(dir_name=cfg.dir['out'] + 'pickle/', file_name='offset_distribution_' + str(t1) + '_' + str(to1) + '_' + str(to2), data=data)

    return mean, var