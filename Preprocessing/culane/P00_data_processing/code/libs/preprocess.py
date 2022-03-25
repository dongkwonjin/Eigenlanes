import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def load_txt_data(self):
        lane_label = dict()
        lane_label['lane_pts_gt'] = list()
        lane_label['exist'] = list()

        with open(self.cfg.dir['dataset'] + 'img/{}.lines.txt'.format(self.img_name[:-4]), 'r') as f:
            lines = f.readlines()
            for j in range(len(lines)):
                if lines[j][-1] == '\n':
                    lines[j] = lines[j][:-2]
                    if lines[j] == '':
                        continue
                    pts = lines[j].split(' ')
                    pts = np.float32([float(p) for p in pts])
                    pts = pts.reshape(2, -1, order='F').transpose(1, 0)
                    lane_label['lane_pts_gt'].append(pts)

            if 'train' in self.cfg.datalist:
                lane_label['exist'] = self.exist

        return lane_label

    def update_data(self, i):
        self.img_name = self.img_list[i]
        if 'train' in self.cfg.datalist:
            self.label_name = self.label_list[i]
            self.exist = self.exist_list[i]
        self.dir_name = os.path.dirname(self.img_name) + '/'
        self.file_name = os.path.basename(self.img_name).replace('.jpg', '')

    def get_datalist(self):
        self.datalist = []
        self.img_list = []
        self.label_list = []
        self.exist_list = []
        with open(self.cfg.dir['dataset'] + 'list/{}.txt'.format(self.cfg.datalist)) as f:
            for line in f:
                data = line.strip().split(" ")
                self.img_list.append(data[0][1:])
                self.datalist.append(data[0][1:].replace('.jpg', ''))
                if 'train' in self.cfg.datalist:
                    self.label_list.append(data[1][1:])
                    self.exist_list.append([int(data[2]), int(data[3]), int(data[4]), int(data[5])])

    def run(self):
        print('start')
        self.get_datalist()

        for i in range(len(self.img_list)):

            self.update_data(i)
            out = self.load_txt_data()

            if self.cfg.save_pickle == True:
                save_pickle(dir_name='{}pickle/{}/'.format(self.cfg.dir['out'], self.dir_name), file_name=self.file_name, data=out)
            print('i : {}, : image name : {} done'.format(i, self.img_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist', data=self.datalist)