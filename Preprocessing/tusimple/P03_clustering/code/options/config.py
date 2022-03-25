import os
import torch

import numpy as np

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_dataloader()
        self.setting_for_visualization()
        self.setting_for_save()
        # --------preprocessing-------- #
        self.setting_for_lane_representation()
        self.setting_for_svd()
        self.setting_for_clustering()
        # --------others-------- #
        self.setting_for_lane_detection()

    def setting_for_system(self):
        self.gpu_id = "1"
        self.seed = 123
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        torch.backends.cudnn.deterministic = True

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()

        self.setting_for_dataset_path()  # dataset path

        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        self.dir['head_proj'] = '/'.join(self.dir['proj'].split('/')[:-2]) + '/'
        self.dir['pre0'] = self.dir['head_proj'] + 'P00_data_processing/output_{}/pickle/'.format(self.datalist)
        self.dir['pre1'] = self.dir['head_proj'] + 'P01_lane_representation/output_{}/pickle/'.format(self.datalist)
        self.dir['pre2'] = self.dir['head_proj'] + 'P02_SVD/output_{}/pickle/'.format(self.datalist)
        self.dir['out'] = os.getcwd().replace('code', 'output') + '_{}/'.format(self.datalist)

    def setting_for_dataset_path(self):
        self.dataset = 'tusimple'  # ['tusimple']
        self.datalist = 'train_set'  # ['train_set'] only

        # ------------------- need to modify -------------------
        self.dir['dataset'] = '--dataset_dir'
        # ------------------------------------------------------

    def setting_for_image_param(self):
        self.org_height = 720
        self.org_width = 1280
        self.height = 384
        self.width = 640
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 180

    def setting_for_dataloader(self):
        self.num_workers = 4
        self.batch_size = 1
        self.data_flip = True

    def setting_for_visualization(self):
        self.display = True
        self.display_all = True
        self.display_mask = True

    def setting_for_save(self):
        self.save_pickle = True

    def setting_for_lane_detection(self):
        self.max_lane_num = 5

    def setting_for_lane_representation(self):
        self.min_y_coord = 0
        self.max_y_coord = 330
        self.node_num = self.max_y_coord
        self.py_coord = self.height - np.float32(np.round(np.linspace(self.max_y_coord, self.min_y_coord + 1, self.node_num)))

    def setting_for_svd(self):
        self.top_m = 4
        self.thresd_iou = 0.7

        # sampling lane component
        self.node_num = 100
        self.sample_idx = np.int32(np.linspace(0, self.max_y_coord - 1, self.node_num))
        self.node_sampling = True
        if self.node_sampling == True:
            self.py_coord = self.py_coord[self.sample_idx]

    def setting_for_clustering(self):
        self.top_m_for_clustering = 4
        self.cluster_mode = 'kmeans'
        self.n_clusters = dict()
        self.n_clusters['straight'] = 300
        self.n_clusters['straight_iter'] = 150
        self.n_clusters['curve'] = 700
        self.n_clusters['curve_iter'] = 350
        self.thresd_theta = 1
        self.thresd_iou = dict()
        self.thresd_iou['straight'] = 0.6
        self.thresd_iou['curve'] = 0.7

        self.scale_factor = [4, 8, 16]
