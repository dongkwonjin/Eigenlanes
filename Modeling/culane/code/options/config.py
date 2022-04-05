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
        self.setting_for_preprocessing()
        # --------modeling-------- #
        self.setting_for_training()
        self.setting_for_evaluation()

    def setting_for_preprocessing(self):
        self.setting_for_lane_representation()
        self.setting_for_svd()
        self.setting_for_clustering()
        self.setting_for_training_label_generation()
        # --------others-------- #
        self.setting_for_lane_detection()

    def setting_for_system(self):
        self.gpu_id = "2"
        self.seed = 123

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()

        self.setting_for_dataset_path()  # dataset path
        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        # ------------------- need to modify ------------------- #
        self.dir['head_pre'] = '--preprocessed data path'
        # ------------------------------------------------------ #
        self.dir['pre0'] = self.dir['head_pre'] + 'P00_data_processing/output_{}/pickle/'.format(self.datalist)
        self.dir['pre1'] = self.dir['head_pre'] + 'P01_lane_representation/output_{}/pickle/'.format(self.datalist)
        self.dir['pre2'] = self.dir['head_pre'] + 'P02_SVD/output_{}/pickle/'.format(self.datalist)
        self.dir['pre3'] = self.dir['head_pre'] + 'P03_clustering/output_{}/pickle/'.format(self.datalist)
        self.dir['pre4'] = self.dir['head_pre'] + 'P04_training_label_generation/output_{}/pickle/'.format(self.datalist)

        self.dir['out'] = os.getcwd().replace('code', 'output') + '/'
        self.dir['weight'] = self.dir['out'] + 'train/weight/'

    def setting_for_dataset_path(self):
        self.dataset_name = 'culane'
        self.datalist = 'train_gt'  # ['train_gt'] only

        # ------------------- need to modify ------------------- #
        self.dir['dataset'] = '--dataset path'
        # ------------------------------------------------------ #

    def setting_for_image_param(self):
        self.org_height = 590
        self.org_width = 1640
        self.height = 320
        self.width = 800
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 240
        self.scale_factor = [8, 16, 32]

    def setting_for_dataloader(self):
        self.num_workers = 4
        self.data_flip = True

        self.category = 'test_all'  # ['test_all']
        self.sampling_step1 = 34
        self.sampling_step2 = 5

        self.batch_size = {'img': 4}

        self.gaussian_blur = True
        self.gaussian_blur_seg = True
        self.kernel_8 = (5, 5)
        self.kernel_16 = (5, 5)

    def setting_for_visualization(self):
        self.disp_step = 50
        self.disp_test_result = False

    def setting_for_save(self):
        self.save_pickle = True

    def setting_for_lane_detection(self):
        self.max_lane_num = 4
        self.constrain_max_score = False
        self.constrain_max_lane_num = True
        self.use_decoder = True

    def setting_for_lane_height(self):
        self.height_class = np.array([35, 52, 75], dtype=np.int32)
        self.height_node_idx = np.array([0, 3, 7], dtype=np.int32)

    def setting_for_lane_representation(self):
        self.min_y_coord = 0
        self.max_y_coord = 285
        self.node_num = self.max_y_coord
        self.py_coord = self.height - np.float32(np.round(np.linspace(self.max_y_coord, self.min_y_coord + 1, self.node_num)))

        self.setting_for_lane_height()

    def setting_for_svd(self):
        self.top_m = 4

        # sampling lane component
        self.node_num = 50
        self.sample_idx = np.int32(np.linspace(0, self.max_y_coord - 1, self.node_num))
        self.node_sampling = True
        if self.node_sampling == True:
            self.py_coord = self.py_coord[self.sample_idx]

    def setting_for_clustering(self):
        self.top_m_for_clustering = 2
        self.n_clusters = 500

    def setting_for_training_label_generation(self):
        return True

    def setting_for_training(self):
        self.run_mode = 'train'  # ['train', 'test', 'eval']
        self.resume = True

        self.lr = 1e-4
        self.milestones = [30, 60, 90, 120, 150, 180, 210]
        self.weight_decay = 5e-4
        self.gamma = 0.5

        self.epochs = 300
        self.epoch_eval = 120
        self.epoch_eval_all = 120
        self.max_iter = 10
        self.max_iter_train = 10
        self.backbone = '18'

        self.setting_for_thresd()

    def setting_for_thresd(self):
        self.thresd_min_offset = -4
        self.thresd_max_offset = 4

        self.thresd_iou_for_cls = 0.6
        self.thresd_iou_for_reg = 0.55

        self.thresd_nms_iou = 0.55
        self.thresd_label_pos_iou = 0.8
        self.thresd_label_adj_iou = 0.5

    def setting_for_evaluation(self):
        self.eval_height = 590
        self.eval_width = 1640

        self.do_eval_culane_official = False
        self.do_eval_culane_laneatt = True

        self.param_name = 'max_laneatt'  # ['trained_last', 'max_laneatt']
