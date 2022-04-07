import cv2
import numpy as np

from libs.utils import *

class Visualize(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)
        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = dict()

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_label(self, label, name='label'):
        label = to_np(label)
        label = np.repeat(np.expand_dims(np.uint8(label != 0) * 255, axis=2), 3, 2)
        self.show[name] = label

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def draw_polyline(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        pts = np.int32(data).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], False, color, s, lineType=cv2.LINE_AA)
        self.show[name] = img

    def display_imglist(self, dir_name, file_name, list):
        disp = self.line
        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)

    def update_datalist(self, img, img_name, label, dir_name, file_name, img_idx):
        self.update_image(img)
        self.update_image_name(img_name)
        self.update_label(label)

        self.dir_name = dir_name
        self.file_name = file_name
        self.img_idx = img_idx

        self.show['img_overlap'] = np.copy(self.show['img'])
        self.show['label_overlap'] = np.copy(self.show['label'])

    def draw_lanes_for_datalist(self, coord, coord_ip, coord_ep, check_ip, check_ep):
        self.draw_polyline(data=coord, name='img_overlap', ref_name='img_overlap', color=(0, 255, 255))
        self.draw_polyline(data=coord_ip[check_ip == 1], name='img_overlap', ref_name='img_overlap', color=(0, 255, 0))
        self.draw_polyline(data=coord_ep[check_ep == 1], name='img_overlap', ref_name='img_overlap', color=(0, 0, 255))
        self.draw_polyline(data=coord_ep[check_ep == 2], name='img_overlap', ref_name='img_overlap', color=(0, 0, 255))
        self.draw_polyline(data=coord_ip[check_ip == 1], name='label_overlap', ref_name='label_overlap', color=(0, 255, 0))
        self.draw_polyline(data=coord_ep[check_ep == 1], name='label_overlap', ref_name='label_overlap', color=(0, 0, 255))
        self.draw_polyline(data=coord_ep[check_ep == 2], name='label_overlap', ref_name='label_overlap', color=(0, 0, 255))

    def save_datalist(self, error_list):
        if error_list[2] == False:
            dir_name = self.cfg.dir['out'] + 'display_all/'
            file_name = str(self.img_idx) + '.jpg'
            self.display_imglist(dir_name, file_name, list=['img', 'img_overlap', 'label_overlap'])
        if error_list[2] == True:
            dir_name = self.cfg.dir['out'] + 'display_error/' + self.dir_name + '/'
            file_name = self.file_name + '.jpg'
            self.display_imglist(dir_name, file_name, list=['img', 'img_overlap', 'label_overlap'])
        if error_list[3] == True:  # short lane
            dir_name = self.cfg.dir['out'] + 'display_error_short/'
            file_name = str(self.img_idx) + '.jpg'
            self.display_imglist(dir_name, file_name, list=['img', 'img_overlap', 'label_overlap'])
