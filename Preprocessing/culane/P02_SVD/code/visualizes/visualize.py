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
        self.update_image(img[0])
        self.update_image_name(img_name)
        self.update_label(label)

        self.dir_name = dir_name
        self.file_name = file_name
        self.img_idx = img_idx

        self.show['img_overlap'] = np.copy(self.show['img'])
        self.show['label_overlap'] = np.copy(self.show['label'])

    def draw_lanes_for_datalist(self, px_coord, px_coord_ap, py_coord):
        for i in range(px_coord.shape[1]):
            node_pts = torch.cat((px_coord[:, i].view(-1, 1), py_coord.view(-1, 1)), dim=1)
            if len(node_pts) == 0:
                continue
            self.draw_polyline(data=to_np(node_pts), name='img_overlap', ref_name='img_overlap', color=(0, 255, 0), s=4)
            self.draw_polyline(data=to_np(node_pts), name='label_overlap', ref_name='label_overlap', color=(0, 255, 0), s=4)
        for i in range(px_coord.shape[1]):
            node_pts = torch.cat((px_coord_ap[:, i].view(-1, 1), py_coord.view(-1, 1)), dim=1)
            if len(node_pts) == 0:
                continue
            self.draw_polyline(data=to_np(node_pts), name='img_overlap', ref_name='img_overlap', color=(0, 0, 255))
            self.draw_polyline(data=to_np(node_pts), name='label_overlap', ref_name='label_overlap', color=(0, 0, 255))

    def save_datalist(self, is_error_case):
        if self.cfg.display_all == True and is_error_case == False:
            dir_name = self.cfg.dir['out'] + 'display/{}/'.format(self.dir_name)
            file_name = '{}.jpg'.format(self.file_name)
            self.display_imglist(dir_name=dir_name, file_name=file_name, list=['img', 'img_overlap', 'label_overlap'])
            dir_name = self.cfg.dir['out'] + 'display_all/'
            file_name = '{}.jpg'.format(str(self.img_idx))
            self.display_imglist(dir_name=dir_name, file_name=file_name, list=['img', 'img_overlap', 'label_overlap'])

        if is_error_case == True:
            dir_name = self.cfg.dir['out'] + 'display_error/{}/'.format(self.dir_name)
            file_name = '{}.jpg'.format(self.file_name)
            self.display_imglist(dir_name=dir_name, file_name=file_name, list=['img', 'img_overlap', 'label_overlap'])

