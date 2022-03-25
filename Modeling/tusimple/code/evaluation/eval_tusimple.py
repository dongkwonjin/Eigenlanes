import torch

from libs.utils import *

import cv2

import numpy as np
from sklearn.linear_model import LinearRegression
import json

def read_json(dirname, fname):
    f = open(dirname + fname)
    data = json.load(f)
    return data

class LaneEval(object):

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.lr = LinearRegression()
        self.pixel_thresh = 20
        self.pt_thresh = 0.85
        self.py_coord = self.cfg.py_coord.reshape(-1, 1).astype(int)

    def get_angle(self, xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            self.lr.fit(ys[:, None], xs)
            k = self.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    def line_accuracy(self, pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    def bench(self, pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [self.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [self.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [self.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < self.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    def get_lane_mask(self, mask, input):
        mask_num = input.shape[0]

        for i in range(mask_num):
            tmp_mask = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
            pts = np.int32(input[i])
            pts = np.concatenate((pts.reshape(-1, 1), self.py_coord), axis=1)
            if self.mode_h == True:
                pts = pts[np.maximum(int(self.height_idx[i]), self.vp_idx):]

            pts[:, 0] = pts[:, 0] / (self.cfg.width - 1) * (self.cfg.org_width - 1)
            pts[:, 1] = pts[:, 1] / (self.cfg.height - 1) * (self.cfg.org_height - self.cfg.crop_size - 1)
            pts[:, 1] += self.cfg.crop_size

            pts[:, 0] = pts[:, 0] / (self.cfg.org_width - 1) * (self.cfg.width - 1)
            pts[:, 1] = pts[:, 1] / (self.cfg.org_height - 1) * (self.cfg.height - 1)

            pts = np.int32(pts).reshape((-1, 1, 2))
            tmp_mask = cv2.polylines(tmp_mask, [pts], False, 255, 1)
            mask[i, :, :] = tmp_mask[:, :]

        return mask

    def prediction(self, data, y_samples):
        out_num = data.shape[0]
        out_mask = np.zeros((out_num, self.cfg.height, self.cfg.width), dtype=np.uint8)
        out_mask = self.get_lane_mask(out_mask, to_np(data))
        pred = []
        for i in range(out_num):
            org_out = cv2.resize(out_mask[i], dsize=(self.cfg.org_width, self.cfg.org_height), interpolation=1)
            tmp = []
            for sample in y_samples:
                if np.sum(org_out[sample, :]) == 0:
                    tmp.append(-2)
                else:
                    tmp.append(np.median(org_out[sample, :].nonzero()[0]))

            pred.append(tmp)
        return pred

    def measure_accuracy(self, mode='test', mode_h = True):
        datalist = load_pickle(self.cfg.dir['out'] + mode + '/pickle/datalist')
        self.mode_h = mode_h
        num = len(datalist)
        total_acc, total_fp, total_fn = 0, 0, 0
        for i in range(num):
            data = load_pickle(self.cfg.dir['out'] + mode + '/pickle/' + datalist[i])
            gt_source = load_pickle(self.cfg.dir['pre0_test'] + datalist[i])
            gt = gt_source['lanes']
            y_samples = gt_source['h_samples']
            out = data['out']['mwcs_reg']
            self.height_idx = data['out']['mwcs_h_idx']

            if data['out']['mwcs_vp_idx'].shape[0] == 0:
                self.vp_idx = 0
            else:
                self.vp_idx = to_np(data['out']['mwcs_vp_idx'])[0]
            out_num = out.shape[0]

            if out_num == 0:
                pred = []
                pred.append(list(np.full(len(y_samples), -2)))
            else:
                pred = self.prediction(out, y_samples)
            acc, fp, fn = self.bench(pred, gt, y_samples, 100)
            total_acc += acc
            total_fp += fp
            total_fn += fn
        total_acc = total_acc/num
        total_fp = total_fp/num
        total_fn = total_fn/num

        print('---------Performance %s---------\n'
              'acc %5f / fp %5f/ fn %5f' % ("tusimple", total_acc, total_fp, total_fn))

        return total_acc, total_fp, total_fn
