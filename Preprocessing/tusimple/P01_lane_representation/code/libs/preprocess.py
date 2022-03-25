import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

from scipy.optimize import curve_fit
from scipy import interpolate

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualize = dict_DB['visualize']

    def get_lane_inf(self, idx):
        if self.flip_idx == 0:
            lane_coord = self.lane_coord[idx][0]
        else:
            lane_coord = self.lane_coord[idx][0]
            lane_coord[:, 0] = self.cfg.org_width - lane_coord[:, 0]

        return to_np2(lane_coord)

    def rescale_coord(self, data):
        data_r = np.copy(data)
        data_r[:, 0] = data_r[:, 0] / (self.cfg.org_width - 1) * (self.cfg.width - 1)
        data_r[:, 1] = (data_r[:, 1] - self.cfg.crop_size) / (self.cfg.org_height - 1 - self.cfg.crop_size) * (self.cfg.height - 1)

        return data_r

    def check_one_to_one_mapping(self, coord):
        dy = (coord[:, 1][1:] - coord[:, 1][:-1])
        c1 = np.sum(dy > 0)
        c2 = np.sum(dy <= 0)

        if c1 * c2 != 0:
            print('error case: not one-to-one mapping! {}'.format(self.img_name))

        return coord

    def compute_angle_of_line_segment(self, x, y):
        angle = np.arctan((y[0] - y[1]) / (x[0] - x[1])) * 180 / np.pi
        return angle

    def lane_interpolation(self, coord):
        self.coord_ip = np.zeros((self.cfg.height + 1, 2), dtype=np.float32)
        self.check_ip = np.zeros((self.cfg.height + 1), dtype=np.int32)
        self.check_la = np.zeros((self.cfg.height + 1), dtype=np.int32)

        self.coord = self.rescale_coord(coord)
        self.coord[:, 1] = np.round(self.coord[:, 1])

        try:
            self.lane_height = np.int32(np.round(np.min(self.coord[:, 1])))
        except:
            print('e')

        if self.coord[0, 1] < self.coord[-1, 1]:
            self.coord = np.flip(self.coord, axis=0)
        coord = np.copy(self.coord)
        num = self.coord.shape[0]

        flag = False

        for i in range(num - 1):
            if flag == True:
                break

            x = coord[i:i + 2, 0]
            y = coord[i:i + 2, 1]

            if y[1] > y[0]:
                y[1] = y[0] - 1
                coord[i + 1, 1] = y[1]

            check_low_angle = False
            angle = self.compute_angle_of_line_segment(x, y)
            if np.abs(angle) < self.cfg.thresd_theta:
                if angle > 0:
                    dy = np.tan(self.cfg.thresd_theta * np.pi / 180) * (x[0] - x[1]) - (y[0] - y[1])
                else:
                    dy = np.tan(-1 * self.cfg.thresd_theta * np.pi / 180) * (x[0] - x[1]) - (y[0] - y[1])

                y[1] = np.int32(np.round(y[1] - dy))
                coord[i+1, 1] = y[1]
                check_low_angle = True

            y_dist = np.int32(np.abs(y[0] - y[1]))
            f = interpolate.interp1d(y, x, kind='linear')
            ynew = np.int32(np.linspace(y[0], y[1], y_dist + 1))
            xnew = np.float32(f(ynew))

            xnew = xnew[(0 <= ynew) * (ynew <= self.cfg.height)]
            ynew = ynew[(0 <= ynew) * (ynew <= self.cfg.height)]

            self.coord_ip[ynew, 0] = xnew
            self.coord_ip[ynew, 1] = ynew
            self.check_ip[ynew] = 1

            if check_low_angle == True:
                self.check_la[ynew] = 1

        self.coord_ip = self.coord_ip[:-1]
        self.check_ip = self.check_ip[:-1]
        self.check_ip[:self.cfg.height - self.cfg.max_y_coord] = 0

        check = (0 <= self.coord_ip[:, 0]) * (self.coord_ip[:, 0] < self.cfg.width)
        check = check * self.check_ip
        coord = self.coord_ip[check == 1, :]
        try:
            dist = np.linalg.norm(coord[0] - coord[-1])
        except:
            dist = 0
        if dist < self.cfg.height // self.cfg.thresd_ratio_for_short_lane:
            print('error case: short lane! {}'.format(self.img_name))
            self.is_short_lane = True
            self.is_error_case4 = True
        if np.sum(self.check_la) > np.sum(self.check_ip) // 3:
            print('error case: a lot of percentage of low angle of lane segment! {}'.format(self.img_name))
            self.is_short_lane = True
            self.is_error_case4 = True

    def func(self, y, a, b):
        return a * y + b

    def func2(self, y, a, b, c):
        return a * y * y + b * y + c

    def lane_extrapolation(self):
        self.coord_ep = np.copy(self.coord_ip)
        self.check_ep = np.zeros((self.cfg.height), dtype=np.int32)

        idx = (self.check_ip == 1).nonzero()[0]

        if self.is_short_lane == True:
            return True

        if idx.shape[0] == 0:
            self.is_error_case_per_lane = True
            print('error case: no inter-interpolation! {}'.format(self.img_name))
            return True

        if np.sum((idx[1:] - idx[:-1]) > 1) != 0:
            self.is_error_case_per_lane = True
            print('error case: not continuous! {}'.format(self.img_name))
            return True

        st = idx[0] - 1
        ed = idx[idx.shape[0] - 1] + 1
        interval = 10

        # vertically upward direction
        for i in range(st, self.cfg.height - 1 - self.cfg.max_y_coord - 1, -1):

            x = self.coord_ep[i + 1:i + interval + 1, 0]
            y = np.round(self.coord_ep[i + 1:i + interval + 1, 1])
            ynew = np.int32([i])

            try:
                popt, pcov = curve_fit(self.func2, y, x)
            except:
                self.is_error_case_per_lane = True
                print('error case: curve fitting! {}'.format(self.img_name))
                break
            xnew = self.func2(ynew, popt[0], popt[1], popt[2])

            self.coord_ep[ynew, 0] = xnew
            self.coord_ep[ynew, 1] = ynew
            self.check_ep[ynew] = 1

        # vertically downward direction
        for i in range(ed, self.cfg.height, 1):

            x = self.coord_ep[i - interval - 1:i, 0]
            y = np.round(self.coord_ep[i - interval - 1:i, 1])
            ynew = np.int32([i])

            try:
                popt, pcov = curve_fit(self.func, y, x)
            except:
                self.is_error_case2 = True
                print('error case: curve fitting! {}'.format(self.img_name))
                break
            xnew = self.func(ynew, popt[0], popt[1])

            self.coord_ep[ynew, 0] = xnew
            self.coord_ep[ynew, 1] = ynew
            self.check_ep[ynew] = 2


    def get_lane_component(self):
        out = {'x_coord': [],
               'height': [],
               'coord_gt': [],
               'check_ip': []}

        k = 0
        for i in range(len(self.lane_coord)):
            self.is_error_case_per_lane = self.is_error_case1
            self.is_short_lane = False

            lane_coord = self.get_lane_inf(k)
            lane_coord = self.check_one_to_one_mapping(lane_coord)

            self.lane_interpolation(lane_coord)
            self.lane_extrapolation()

            if self.cfg.display_all == True or self.is_error_case_per_lane == True or self.is_short_lane == True:
                self.visualize.draw_lanes_for_datalist(self.coord, self.coord_ip, self.coord_ep, self.check_ip, self.check_ep)

            if self.is_error_case_per_lane == False and self.is_short_lane == False:
                x_coord = self.coord_ep[:, 0][np.int32(self.cfg.py_coord)]
                out['x_coord'].append(x_coord)
                out['height'].append(self.lane_height)
                idx = (self.check_ip == 1).nonzero()[0]
                out['check_ip'].append(self.check_ip)
                check = (self.coord_ip[idx][:, 1] >= self.lane_height - 2)
                out['coord_gt'].append(self.coord_ip[idx][check, :])
                self.height_distribution[self.lane_height] += 1

            k += 1

            if self.is_error_case_per_lane == True:
                self.is_error_case2 = True

        return out

    def get_flipped_data(self, pre_out):
        out = {'x_coord': [],
               'height': [],
               'coord_gt': [],
               'check_ip': []}
        for i in range(len(pre_out['x_coord']) - 1, -1, -1):
            x_coord = self.cfg.width - 1 - pre_out['x_coord'][i]

            refined_coord = np.copy(pre_out['coord_gt'][i])
            refined_coord[:, 0] = self.cfg.width - 1 - refined_coord[:, 0]

            out['x_coord'].append(x_coord)
            out['coord_gt'].append(refined_coord)
            out['height'].append(pre_out['height'][i])
            out['check_ip'].append(pre_out['check_ip'][i])

        return out

    def run_flip(self):

        for i in range(0, 2):  # 1: horizontal flip
            self.flip_idx = i

            if i == 1:
                self.img = self.img.flip(2)
                self.label = self.label.flip(1)

            self.visualize.update_datalist(self.img, self.img_name, self.label, self.dir_name, self.file_name, self.img_idx)

            if i == 0:
                self.out_f.append(self.get_lane_component())
                self.is_error_case3 = bool(self.is_error_case1 + self.is_error_case2)
            elif self.is_error_case3 == False:
                self.out_f.append(self.get_flipped_data(self.out_f[0]))

            if i == 0 and self.cfg.display_all == True:
                self.visualize.save_datalist([self.is_error_case1, self.is_error_case2, self.is_error_case3, self.is_error_case4])

    def update_batch_data(self, batch, i):
        self.img = batch['img'][0].cuda()
        self.label = batch['label'][0].cuda()
        self.img_name = batch['img_name'][0].replace('.jpg', '')
        self.img_idx = i
        self.dir_name = os.path.dirname(self.img_name)
        self.file_name = os.path.basename(self.img_name)

        self.lane_coord = batch['lane_pts']

        self.out_f = list()
        self.is_error_case1 = False
        self.is_error_case2 = False
        self.is_error_case3 = False
        self.is_error_case4 = False

    def init(self):
        self.datalist = list()
        self.datalist_error = list()
        self.height_distribution = np.zeros((self.cfg.height), dtype=np.int64)

    def run(self):
        print('start')

        self.init()

        for i, batch in enumerate(self.dataloader):
            self.update_batch_data(batch, i)
            self.run_flip()

            if self.cfg.save_pickle == True:
                save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name=self.img_name, data=self.out_f)
                if self.is_error_case3 == False:
                    self.datalist.append(self.img_name)
                else:
                    self.datalist_error.append(self.img_name)

            print('image {} ===> {} clear'.format(i, self.img_name))

        if self.cfg.save_pickle == True:
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'datalist', self.datalist)
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'datalist_error', self.datalist_error)
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'height_distribution', self.height_distribution)
