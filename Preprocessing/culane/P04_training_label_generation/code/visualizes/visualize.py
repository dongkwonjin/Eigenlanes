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

        self.show = {}

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
        img = cv2.polylines(img, [pts], False, color, s)
        self.show[name] = img

    def display_imglist(self, dir_name, file_name, list):
        disp = self.line
        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)

    def update_datalist(self, img, img_name, label):
        self.update_image(img)
        self.update_image_name(img_name)
        self.update_label(label)

        self.show['img_overlap'] = np.copy(self.show['img'])
        self.show['label_overlap'] = np.copy(self.show['label'])
        self.show['label_overlap_reg'] = np.copy(self.show['label'])

    def draw_lanes_for_datalist(self, temp, U, org_lane, py_coord):
        for k in range(len(org_lane['x_coord'])):
            if len(org_lane['x_coord'][k]) == 0:
                continue
            cand_px_coord = temp['cand_px_coord'][k]
            cand_c = temp['cand_c'][k]
            px_coord = temp['org_px_coord'][k]
            offset = temp['offset'][k]

            reg_c = (cand_c + offset).permute(1, 0)
            reg_px_coord = torch.matmul(U[:, :self.cfg.top_m], reg_c).permute(1, 0) * (self.cfg.width - 1)
            for l in range(cand_px_coord.shape[0]):
                node_pts = torch.cat((cand_px_coord[l, :].view(-1, 1), py_coord.view(-1, 1)), dim=1)
                self.draw_polyline(data=to_np(node_pts), name='img_overlap', ref_name='img_overlap', color=(0, 255, 0))
                self.draw_polyline(data=to_np(node_pts), name='img_overlap', ref_name='img_overlap', s=2, color=(0, 255, 0))
                self.draw_polyline(data=to_np(node_pts), name='label_overlap', ref_name='label_overlap', color=(0, 255, 0))
                self.draw_polyline(data=to_np(node_pts), name='label_overlap', ref_name='label_overlap', s=2, color=(0, 255, 0))
            for l in range(reg_px_coord.shape[0]):
                node_pts = torch.cat((reg_px_coord[l, :].view(-1, 1), py_coord.type(torch.float).view(-1, 1)), dim=1)
                self.draw_polyline(data=to_np(node_pts), name='label_overlap_reg', ref_name='label_overlap_reg', color=(0, 255, 0))
                self.draw_polyline(data=to_np(node_pts), name='label_overlap_reg', ref_name='label_overlap_reg', s=2, color=(0, 255, 0))

            node_pts = torch.cat((px_coord.view(-1, 1), py_coord.view(-1, 1)), dim=1)
            self.draw_polyline(data=to_np(node_pts), name='img_overlap', ref_name='img_overlap', color=(0, 0, 255))
            self.draw_polyline(data=to_np(node_pts), name='img_overlap', ref_name='img_overlap', s=2, color=(0, 0, 255))
            self.draw_polyline(data=to_np(node_pts), name='label_overlap', ref_name='label_overlap', color=(0, 0, 255))
            self.draw_polyline(data=to_np(node_pts), name='label_overlap', ref_name='label_overlap', s=2, color=(0, 0, 255))

    def save_datalist(self, is_error_case, idx):
        if is_error_case == 0:
            dir_name = self.cfg.dir['out'] + 'display/'
            file_name = str(idx) + '.jpg'
            self.display_imglist(dir_name=dir_name, file_name=file_name, list=['img', 'img_overlap', 'label_overlap', 'label_overlap_reg'])
        elif is_error_case == 1:
            dir_name = self.cfg.dir['out'] + 'display_error_low_iou/'
            file_name = str(idx) + '.jpg'
            self.display_imglist(dir_name=dir_name, file_name=file_name, list=['img', 'img_overlap', 'label_overlap', 'label_overlap_reg'])
        elif is_error_case == 2:
            dir_name = self.cfg.dir['out'] + 'display_error_large_offset/'
            file_name = str(idx) + '.jpg'
            self.display_imglist(dir_name=dir_name, file_name=file_name, list=['img', 'img_overlap', 'label_overlap', 'label_overlap_reg'])

