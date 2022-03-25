import cv2
import shutil
import torch

import numpy as np

from libs.utils import *

class Save_Prediction(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def make_file(self, name):
        line_save_path = self.output_dir + name + '.lines.txt'
        self.line_save_path = line_save_path
        save_dir, _ = os.path.split(line_save_path)
        mkdir(save_dir)

    def write_data(self):
        with open(self.line_save_path, 'w') as f:
            for i in range(len(self.pred_xy_coord)):
                for j in range(self.pred_xy_coord[i].shape[0]):
                    f.write('%d %d ' % (int(self.pred_xy_coord[i][j][0]), int(self.pred_xy_coord[i][j][1])))
                f.write('\n')

    def rescale_pts(self, data, mode):
        eval_width = self.cfg.eval_width
        eval_height = self.cfg.eval_height
        org_height = self.cfg.org_height
        org_width = self.cfg.org_width
        height = self.cfg.height
        width = self.cfg.width
        crop_size = self.cfg.crop_size

        if data.shape[0] != 0:
            if mode == 'width':
                data = data * (eval_width - 1) / (width - 1)
            else:
                data = data * (org_height - crop_size - 1) / (height - 1)
                data += crop_size
                data = data / (org_height - 1) * eval_height
        return data

    def get_2D_lane_points(self, px_coord):
        x = []
        y = []
        px_coord = self.rescale_pts(px_coord, mode='width')
        py_coord = self.rescale_pts(self.py_coord, mode='height')
        for i in range(px_coord.shape[0]):
            x.append(px_coord[i])
            y.append(py_coord[0])

        self.data_x = np.array(x)
        self.data_y = np.array(y)

        self.pred_xy_coord = list()
        for i in range(self.data_x.shape[0]):
            height_idx = np.maximum(int(self.vp_idx + 1), int(min(self.h_cls_idx)))
            data_x = self.data_x[i, height_idx:]
            data_y = self.data_y[i, height_idx:]
            data_xy = np.concatenate((data_x[:, np.newaxis], data_y[:, np.newaxis]), axis=1)
            self.pred_xy_coord.append(data_xy)

    def load_pred_data(self):
        data = load_pickle(self.cfg.dir['out'] + self.test_mode + '/pickle/' + self.file_name)
        self.out = data['out']
        if self.use_reg == True:
            self.out_x_coord = self.out[self.key[0] + '_reg']
        else:
            self.out_x_coord = self.out[self.key[0]]

        if self.use_height_cls == True:
            self.h_cls_idx = self.out[self.key[0] + '_h_idx']
        else:
            self.h_cls_idx = np.zeros((self.out_x_coord.shape[0]), dtype=np.int32)
        self.vp_idx = self.out[self.key[0] + '_vp_idx']

        self.get_2D_lane_points(to_np(self.out_x_coord))

    def write_pred_data(self):
        self.make_file(self.file_name)
        self.write_data()

    def run(self):
        self.datalist = load_pickle(self.cfg.dir['out'] + self.test_mode + '/pickle/datalist')

        for i in range(len(self.datalist)):
            self.file_name = self.datalist[i].replace('.jpg', '')
            self.load_pred_data()
            self.write_pred_data()

    def settings(self, key, test_mode='test', use_reg=True, use_height_cls=True):
        self.key = key
        self.test_mode = test_mode
        self.use_reg = use_reg
        self.use_height_cls = use_height_cls
        self.py_coord = np.float32(self.cfg.py_coord).reshape(1, -1)

        self.output_dir = self.cfg.dir['out'] + test_mode + '/results/'
        mkdir(self.output_dir)
