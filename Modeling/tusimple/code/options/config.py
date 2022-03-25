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
        self.gpu_id = "1"
        self.seed = 123

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()

        self.setting_for_dataset_path()  # dataset path

        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        # ------------------- need to modify ------------------- #
        self.dir['head_pre'] = '--preprocessed data path'
        # ------------------------------------------------------ #
        self.dir['pre0_train'] = self.dir['head_pre'] + 'P00_data_processing/output_train_set/pickle/'
        self.dir['pre0_test'] = self.dir['head_pre'] + 'P00_data_processing/output_test_set/pickle/'
        self.dir['pre1'] = self.dir['head_pre'] + 'P01_lane_representation/output_{}/pickle/'.format(self.datalist)
        self.dir['pre2'] = self.dir['head_pre'] + 'P02_SVD/output_{}/pickle/'.format(self.datalist)
        self.dir['pre3'] = self.dir['head_pre'] + 'P03_clustering/output_{}/pickle/'.format(self.datalist)
        self.dir['pre4'] = self.dir['head_pre'] + 'P04_training_label_generation/output_{}/pickle/'.format(self.datalist)

        self.dir['out'] = os.getcwd().replace('code', 'output') + '/'
        self.dir['weight'] = self.dir['out'] + 'train/weight/'

    def setting_for_dataset_path(self):
        self.dataset_name = 'tusimple'  # ['tusimple']
        self.datalist = 'train_set'  # ['train_set'] only

        # ------------------- need to modify ------------------- #
        self.dir['dataset'] = '--dataset path'
        # ------------------------------------------------------ #

    def setting_for_image_param(self):
        self.org_height = 720
        self.org_width = 1280
        self.height = 384
        self.width = 640
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 180
        self.scale_factor = [8, 16, 32]

    def setting_for_dataloader(self):
        self.num_workers = 4
        self.data_flip = True

        self.sampling = False
        self.sampling_step = 1

        self.batch_size = {'img': 8}

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
        self.max_lane_num = 5
        self.constrain_max_score = False
        self.constrain_max_lane_num = True
        self.use_decoder = True

    def setting_for_lane_height(self):
        self.height_class = np.array([57, 63, 70], dtype=np.int32)
        self.height_node_idx = np.array([1, 3, 5], dtype=np.int32)

    def setting_for_lane_representation(self):
        self.min_y_coord = 0
        self.max_y_coord = 330
        self.node_num = self.max_y_coord
        self.py_coord = self.height - np.float32(np.round(np.linspace(self.max_y_coord, self.min_y_coord + 1, self.node_num)))

        self.setting_for_lane_height()

    def setting_for_svd(self):
        self.top_m = 4

        # sampling lane component
        self.node_num = 100
        self.sample_idx = np.int32(np.linspace(0, self.max_y_coord - 1, self.node_num))
        self.node_sampling = True
        if self.node_sampling == True:
            self.py_coord = self.py_coord[self.sample_idx]

    def setting_for_clustering(self):
        self.top_m_for_clustering = 4
        self.cluster_mode = 'kmeans'
        self.n_clusters = 1000

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
        self.epoch_eval = 60
        self.epoch_eval_all = 60
        self.max_iter = 8
        self.max_iter_train = 8
        self.backbone = '18'

        self.setting_for_thresd()

    def setting_for_thresd(self):
        self.thresd_min_offset = -4
        self.thresd_max_offset = 4

        self.thresd_iou_for_cls = 0.6
        self.thresd_iou_upper_for_cls = 0.75

        self.thresd_iou_for_reg = 0.5
        self.thresd_iou_upper_for_reg = 0.65

        self.thresd_score = -1

        self.thresd_nms_iou = 0.65
        self.thresd_nms_iou_upper = 0.75
        self.thresd_label_pos_iou = 0.75
        self.thresd_label_adj_iou = 0.55

    def setting_for_evaluation(self):
        self.param_name = 'max'  # ['trained_last', 'max']
