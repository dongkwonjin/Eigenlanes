import cv2
import math
import time

import torch
import torch.nn.functional as F

from libs.utils import *
from sklearn.cluster import k_means

class Preprocessing(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.visualize = dict_DB['visualize']

    def clustering(self):
        U = load_pickle(self.cfg.dir['pre2'] + 'U')
        S = load_pickle(self.cfg.dir['pre2'] + 'S')

        if self.cfg.save_pickle == True:
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'U', U)
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'S', S)

        n_clusters = self.cfg.n_clusters[self.shape + '_iter']
        data = self.data[:self.cfg.top_m_for_clustering, :]
        X = to_np(data.permute(1, 0))

        ftime = 0
        t = time.time()

        # do clustering
        if self.cfg.cluster_mode == 'kmeans':
            print('do clustering')
            cluster_result = k_means(X, n_clusters=n_clusters)
            if self.cfg.save_pickle == True:
                save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name=self.cfg.cluster_mode, data=cluster_result)
            cluster_result = load_pickle(self.cfg.dir['out'] + 'pickle/' + self.cfg.cluster_mode)
            candidates = cluster_result[0]  # determine centroids as lane candidates
        ftime += (time.time() - t)
        print("Clustering done! Samples: {}, Centers: {} at {} mins".format(X.shape[0], n_clusters, ftime / 60))

        U = U[:, :self.cfg.top_m_for_clustering]
        c = to_tensor(candidates).permute(1, 0)

        px_coord = torch.matmul(U, c) * (self.cfg.width - 1)
        px_coord = px_coord.permute(1, 0)
        py_coord = to_tensor(self.cfg.py_coord).type(torch.float)

        if self.cfg.display == True:
            for i in range(candidates.shape[0]):
                node_pts = torch.cat((px_coord[i, :].view(-1, 1), py_coord.view(-1, 1)), dim=1)
                self.visualize.draw_polyline(data=to_np(node_pts), name='candidates_' + self.shape, ref_name='candidates_' + self.shape, color=(0, 255, 0))

        buffer = to_tensor(np.zeros((self.cfg.top_m - self.cfg.top_m_for_clustering, n_clusters), dtype=np.float32))
        c = torch.cat((c, buffer), dim=0)
        return {'px_coord': px_coord, 'c': c.permute(1, 0)}

    def compute_theta_between_two_line_segment(self, x1, y1, x2, y2, x3, y3):
        v1 = torch.DoubleTensor([x1 - x2, y1 - y2])
        v2 = torch.DoubleTensor([x3 - x2, y3 - y2])

        t1 = torch.acos(torch.sum(v1 * v2) / (torch.norm(v1) * torch.norm(v2)))
        t2 = torch.acos(torch.sum(v1 * -1 * v2) / (torch.norm(v1) * torch.norm(v2)))

        theta = torch.min(t1, t2) / math.pi * 180

        return theta

    def check_lane_curveness(self, x_coord, y_coord, s1, s2, s3):
        num = x_coord.shape[0]

        k1 = int((num - 1) / s1)
        k2 = int((num - 1) / s2)
        k3 = int((num - 1) / s3)

        theta = self.compute_theta_between_two_line_segment(x1=x_coord[k1], y1=y_coord[k1],
                                                            x2=x_coord[k2], y2=y_coord[k2],
                                                            x3=x_coord[k3], y3=y_coord[k3])

        if theta < self.cfg.thresd_theta:
            check = 0
        else:
            check = 1
        return theta, check

    def load_data(self):
        datalist = load_pickle(self.cfg.dir['pre2'] + 'datalist')

        for i in range(len(datalist)):
            img_name = datalist[i]
            data = load_pickle(self.cfg.dir['pre2'] + img_name)
            for j in range(0, 2):  # 1: horizontal flip
                if j == 1 and self.cfg.data_flip == False:
                    continue
                if j == 1 and self.cfg.datalist == 'test':
                    break
                data_c = data[j]['c']
                data_x = data[j]['px_coord_ap']

                if len(data_c) == 0:
                    continue
                for k in range(data_c.shape[1]):
                    c = to_tensor(data_c[:, k])
                    x_coord = to_tensor(data_x[:, k])

                    if self.shape == 'all':
                        self.data = torch.cat((self.data, c.view(-1, 1)), dim=1)
                        self.data_x = torch.cat((self.data_x, x_coord.view(-1, 1)), dim=1)
                    else:
                        y_coord = to_tensor(self.cfg.py_coord)
                        _, curve_check = self.check_lane_curveness(x_coord, y_coord, s1=x_coord.shape[0], s2=2, s3=1)
                        if self.shape == 'straight' and curve_check == 0:
                            self.data = torch.cat((self.data, c.view(-1, 1)), dim=1)
                            self.data_x = torch.cat((self.data_x, x_coord.view(-1, 1)), dim=1)
                        elif self.shape == 'curve' and curve_check == 1:
                            self.data = torch.cat((self.data, c.view(-1, 1)), dim=1)
                            self.data_x = torch.cat((self.data_x, x_coord.view(-1, 1)), dim=1)

            if i % 100 == 0:
                print('load {} {} done'.format(i, img_name))

        print('The number of {} lanes : {}'.format(self.shape, self.data.shape[1]))

        if self.cfg.save_pickle == True:
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'coefficient_vector_' + self.shape, self.data)
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'x_coord_' + self.shape, self.data_x)

    def generate_candidate_mask(self, out):
        px_coord = out['px_coord']
        py_coord = to_tensor(self.cfg.py_coord).type(torch.float)

        candidate_mask = dict()
        for sf in self.cfg.scale_factor:
            mask_temp = torch.FloatTensor([]).cuda()
            px_coord_s = px_coord / (self.cfg.width - 1) * (self.cfg.width // sf - 1)
            py_coord_s = py_coord / (self.cfg.height - 1) * (self.cfg.height // sf - 1)

            for i in range(px_coord.shape[0]):
                self.visualize.show['mask'] = np.zeros((self.cfg.height // sf, self.cfg.width // sf), dtype=np.uint8)
                node_pts = torch.cat((px_coord_s[i, :].view(-1, 1), py_coord_s.view(-1, 1)), dim=1)

                if sf == 4:
                    self.visualize.draw_polyline(data=to_np(node_pts), name='mask', ref_name='mask', color=1, s=10)
                else:
                    self.visualize.draw_polyline(data=to_np(node_pts), name='mask', ref_name='mask', color=1, s=1)

                if np.sum(self.visualize.show['mask']) == 0:
                    print('error')

                if self.cfg.display_mask == True:
                    dir_name = self.cfg.dir['out'] + 'display/mask/'
                    file_name = 'mask_{}.jpg'.format(str(i))
                    mkdir(dir_name)
                    cv2.imwrite(dir_name + file_name, to_3D_np(self.visualize.show['mask']) * 255)

                mask = to_tensor(self.visualize.show['mask']).unsqueeze(0).type(torch.float)
                mask_temp = torch.cat((mask_temp, mask), dim=0)

            candidate_mask[sf] = mask_temp

        return candidate_mask

    def measure_IoU(self, X1, X2):
        X = X1 + X2

        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

        iou = X_inter / X_uni

        return iou

    def compute_iou_based_iou_map(self, out, out_mask):
        cand_num = out['px_coord'].shape[0]
        iou_map = torch.FloatTensor([]).cuda()
        iou_upper_map = torch.FloatTensor([]).cuda()
        cand_mask = out_mask[4]

        for i in range(cand_num):
            mask = cand_mask[i:i + 1]
            h = mask.shape[1]
            h2 = int((self.cfg.py_coord[0] + self.cfg.max_y_coord / 3) // 4)
            iou = self.measure_IoU(mask[:, :h], cand_mask[:, :h])
            iou_upper = self.measure_IoU(mask[:, :h2], cand_mask[:, :h2])
            nan_idx = torch.isnan(iou_upper).nonzero()[:, 0]
            iou_upper[nan_idx] = 1
            if nan_idx.shape[0] > 0:
                print('solve nan error')
            if np.sum(np.isnan(to_np(iou_upper))) > 0:
                print('occur nan error')

            iou_map = torch.cat((iou_map, iou.view(1, -1)), dim=0)
            iou_upper_map = torch.cat((iou_upper_map, iou_upper.view(1, -1)), dim=0)

        return to_np(iou_map), to_np(iou_upper_map)

    def get_lane_mask(self, data_x, data_y, sf, s):
        temp = np.zeros((self.cfg.height // sf, self.cfg.width // sf), dtype=np.float32)
        temp = np.ascontiguousarray(temp)

        x = data_x / (self.cfg.width - 1) * (self.cfg.width // sf - 1)
        y = data_y / (self.cfg.height - 1) * (self.cfg.height // sf - 1)

        xy_coord = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        xy_coord = np.int32(xy_coord).reshape((-1, 1, 2))

        lane_mask = cv2.polylines(temp, [xy_coord], False, 1, s)
        return to_tensor(lane_mask).unsqueeze(0)

    def get_batch_lane_mask(self, data, key=None):
        out_mask = torch.FloatTensor([]).cuda()
        for i in range(data.shape[0]):
            mask = self.get_lane_mask(to_np(data[i]), self.cfg.py_coord, sf=8, s=2)
            out_mask = torch.cat((out_mask, mask), dim=0)

            if i % 100 == 0:
                print('{} lane mask generation {} done!'.format(key[0], i))

        return {key[0]: out_mask}

    def compute_iou_map(self, data1, data2):
        num1 = data1.shape[0]
        num2 = data2.shape[0]

        iou_map = torch.FloatTensor([]).cuda()

        for i in range(num1):
            mask = data1[i:i + 1]
            _, h, w = mask.shape
            iou = self.measure_IoU(mask.view(1, h, w), data2)
            nan_idx = torch.isnan(iou).nonzero()[:, 0]
            iou[nan_idx] = 0
            if nan_idx.shape[0] > 0:
                print('solve nan error')

            iou_map = torch.cat((iou_map, iou.view(1, -1)), dim=0)

        return {'iou_map': to_np(iou_map)}

    def check_overlapping_data_samples(self, data):
        check = np.sum(data > self.cfg.thresd_iou[self.shape], axis=0)
        return {'check_removal': check}

    def run(self):
        print('start')

        shape_list = ['straight']

        out_f = dict()
        out_f['px_coord'] = torch.FloatTensor([]).cuda()
        out_f['c'] = torch.FloatTensor([]).cuda()

        for shape in shape_list:
            self.shape = shape
            temp = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8) + 100
            self.visualize.show['candidates_' + self.shape] = np.copy(temp)

            self.data = torch.FloatTensor([]).cuda()
            self.data_x = torch.FloatTensor([]).cuda()
            self.load_data()
            self.data = load_pickle(self.cfg.dir['out'] + 'pickle/coefficient_vector_' + self.shape)
            self.data_x = load_pickle(self.cfg.dir['out'] + 'pickle/x_coord_' + self.shape)

            if self.data.shape[1] > self.cfg.thresd_n_sample:
                sample_idx = np.int32(np.linspace(0, self.data.shape[1] - 1, self.cfg.thresd_n_sample))
                self.data = self.data[:, sample_idx]
                self.data_x = self.data_x[:, sample_idx]

            out = dict()
            out['px_coord'] = torch.FloatTensor([]).cuda()
            out['c'] = torch.FloatTensor([]).cuda()

            lane_mask = dict()
            lane_mask.update(self.get_batch_lane_mask(self.data_x.permute(1, 0), key=['sample']))
            save_pickle(self.cfg.dir['out'] + 'pickle/', 'lane_mask_sample', lane_mask)

            lane_mask = load_pickle(self.cfg.dir['out'] + 'pickle/lane_mask_sample')
            for iter in range(self.cfg.n_clusters[shape] // self.cfg.n_clusters[self.shape + '_iter']):
                out_iter = self.clustering()
                lane_mask.update(self.get_batch_lane_mask(out_iter['px_coord'], key=['centroid']))
                out_iter.update(self.compute_iou_map(lane_mask['centroid'], lane_mask['sample']))
                out_iter.update(self.check_overlapping_data_samples(out_iter['iou_map']))

                # removal
                self.data_x = self.data_x[:, out_iter['check_removal'] == 0]
                self.data = self.data[:, out_iter['check_removal'] == 0]
                lane_mask['sample'] = lane_mask['sample'][out_iter['check_removal'] == 0]

                out['px_coord'] = torch.cat((out['px_coord'], out_iter['px_coord']), dim=0)
                out['c'] = torch.cat((out['c'], out_iter['c']), dim=0)

                print('{} clustering process iteration : {} done!'.format(shape, iter))

                if self.cfg.display_all == True:
                    dir_name = self.cfg.dir['out'] + 'display/'
                    file_name = 'candidates_{}_{}_{}.jpg'.format(self.shape, str(self.cfg.n_clusters[self.shape]), str(iter))
                    self.visualize.display_imglist(dir_name=dir_name, file_name=file_name, list=['candidates_iter_' + self.shape,
                                                                                                 'remained_' + self.shape,
                                                                                                 'candidates_' + self.shape])
                if self.cfg.display_all == True:
                    dir_name = self.cfg.dir['out'] + 'display/'
                    file_name = 'candidates_{}_{}.jpg'.format(self.shape, str(self.cfg.n_clusters[self.shape]))
                    self.visualize.display_imglist(dir_name=dir_name, file_name=file_name, list=['candidates_' + self.shape])

            out_f['px_coord'] = torch.cat((out_f['px_coord'], out['px_coord']), dim=0)
            out_f['c'] = torch.cat((out_f['c'], out['c']), dim=0)

            if self.cfg.display_all == True:
                dir_name = self.cfg.dir['out'] + 'display/'
                file_name = 'candidates_{}_{}.jpg'.format(self.shape, str(self.cfg.n_clusters[self.shape]))
                self.visualize.display_imglist(dir_name=dir_name, file_name=file_name, list=['candidates_' + self.shape])

        out_f['c'] = to_np(out_f['c'])
        out_mask = self.generate_candidate_mask(out_f)
        out_iou_map, out_iou_upper_map = self.compute_iou_based_iou_map(out_f, out_mask)

        if self.cfg.save_pickle == True:
            num = 0
            for shape in shape_list:
                num += self.cfg.n_clusters[shape]

            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='lane_candidates_' + str(num), data=out_f)
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='candidate_mask_' + str(num), data=out_mask)
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='candidate_iou_map_' + str(num), data=out_iou_map)
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='candidate_iou_upper_map_' + str(num), data=out_iou_upper_map)
