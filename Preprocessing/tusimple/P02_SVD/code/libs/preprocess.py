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

    def get_lane_mask(self, px_coord, sf=4, s=3):

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

    def compute_approximation_error(self, px_coord_ap, px_coord):
        for i in range(px_coord_ap.shape[1]):
            lane_mask1 = self.get_lane_mask(to_np(px_coord_ap[:, i]))
            lane_mask2 = self.get_lane_mask(to_np(px_coord[:, i]))

            iou = self.measure_IoU(lane_mask1, lane_mask2)

            if iou < self.cfg.thresd_iou:
                print('approximation error : IoU : {}'.format(iou))
                self.is_error_case = True
                break

    def approximate_lane(self):
        out = self.data[self.flip_idx]

        px_coord = torch.FloatTensor([]).cuda()
        for i in range(len(out['x_coord'])):
            x_coord = out['x_coord'][i]
            if len(x_coord) == 0:
                continue
            px_coord = torch.cat((px_coord, to_tensor(x_coord[self.cfg.sample_idx]).view(-1, 1)), dim=1)

        if px_coord.shape[0] == 0:
            return {'px_coord_ap': [], 'c': []}

        U = self.U[:, :self.cfg.top_m]
        U_t = self.U[:, :self.cfg.top_m].permute(1, 0)
        px_coord = px_coord.type(torch.float)

        c = torch.matmul(U_t, (px_coord / (self.cfg.width - 1)))
        px_coord_ap = torch.matmul(U, c) * ((self.cfg.width - 1))
        py_coord = to_tensor(self.cfg.py_coord).type(torch.float)

        self.compute_approximation_error(px_coord_ap, px_coord)

        if self.cfg.display_all == True or self.is_error_case == True:
            self.visualize.draw_lanes_for_datalist(px_coord, px_coord_ap, py_coord)

        return {'px_coord_ap': px_coord_ap, 'c': c}

    def construct_lane_matrix(self):
        datalist = load_pickle(self.cfg.dir['pre1'] + 'datalist')

        self.mat = torch.FloatTensor([]).cuda()
        for i in range(len(datalist)):
            img_name = datalist[i]
            data = load_pickle(self.cfg.dir['pre1'] + img_name)

            for j in range(0, 2):  # 1: horizontal flip
                if j == 1 and self.cfg.data_flip == False:
                    continue
                if j == 1 and self.cfg.datalist == 'test':
                    break
                for k in range(len(data[j]['x_coord'])):
                    if len(data[j]['x_coord'][k]) == 0:
                        continue
                    x_coord = data[j]['x_coord'][k]
                    x_data = to_tensor(x_coord)
                    self.mat = torch.cat((self.mat, x_data.view(-1, 1)), dim=1)

            print('%d done!' % i)

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='matrix', data=self.mat)

    def do_SVD(self):
        U, S, V = torch.svd((self.mat.cpu() / (self.cfg.width - 1)))
        self.U = U.cuda()
        self.S = S.cuda()
        self.V = V.cuda()

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='U', data=self.U)
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='S', data=self.S)

    def run_flip(self):
        for i in range(0, 2):  # 1: horizontal flip
            if i == 1 and self.cfg.data_flip == False:
                continue
            if i == 1 and self.cfg.datalist == 'test':
                break
            self.flip_idx = i

            if i == 1:
                self.img = self.img.flip(2)
                self.label = self.label.flip(1)

            self.visualize.update_datalist(self.img, self.img_name, self.label, self.dir_name, self.file_name, self.img_idx)
            self.out_f.append(self.approximate_lane())
            # save data
            self.visualize.save_datalist(self.is_error_case)

    def update_batch_data(self, batch, i):
        self.img = batch['img'][0].cuda()
        self.label = batch['label'][0].cuda()
        self.img_name = batch['img_name'][0].replace('jpg', '')
        self.img_idx = i
        self.dir_name = os.path.dirname(self.img_name)
        self.file_name = os.path.basename(self.img_name)

        self.data = load_pickle(self.cfg.dir['pre1'] + self.img_name)

        self.out_f = list()
        self.is_error_case = False

    def run(self):
        print('start')

        datalist = []
        datalist_error = []

        self.construct_lane_matrix()

        self.mat = load_pickle(self.cfg.dir['out'] + 'pickle/matrix')
        if self.cfg.node_sampling == True:
            self.mat = self.mat[self.cfg.sample_idx]

        self.do_SVD()

        self.U = load_pickle(self.cfg.dir['out'] + 'pickle/U')
        self.S = load_pickle(self.cfg.dir['out'] + 'pickle/S')

        for i, batch in enumerate(self.dataloader):
            self.update_batch_data(batch, i)
            self.run_flip()

            if self.cfg.save_pickle == True:
                save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name=self.img_name, data=self.out_f)
                if self.is_error_case == False:
                    datalist.append(self.img_name)
                else:
                    datalist_error.append(self.img_name)

            print('image {} ===> {} clear'.format(i, self.img_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist', data=datalist)
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist_error', data=datalist_error)