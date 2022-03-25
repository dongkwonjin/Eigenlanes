import cv2
import ast

import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):

    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

    def run(self):

        out_f = dict()
        out_f['datalist'] = list()

        print('start')

        json_list = dict()
        json_list['train_set'] = ['label_data_0313', 'label_data_0531', 'label_data_0601']
        json_list['test_set'] = ['test_label']

        mode = self.cfg.datalist
        data = list()
        if mode == 'train_set':
            for name in json_list[mode]:
                data += [json.loads(line) for line in open(self.cfg.dir['dataset'] + mode + '/' + name + '.json', 'r')]
        if mode == 'test_set':
            for name in json_list[mode]:
                data += [json.loads(line) for line in open(self.cfg.dir['dataset'] + mode + '/' + name + '.json', 'r')][0][:-1]
            for i in range(len(data)):
                data[i] = ast.literal_eval(data[i])

        self.datalist = list()
        for i in range(len(data)):
            data_s = data[i]
            data_name = data_s['raw_file'].replace('.jpg', '').replace('clips/', '')

            dir_name = self.cfg.dir['out'] + 'pickle/' + '/'.join(data_name.split('/')[:-1]) + '/'
            file_name =data_name.split('/')[-1]
            if self.cfg.save_pickle == True:
                save_pickle(dir_name=dir_name, file_name=file_name, data=data_s)

            self.datalist.append(data_name)

            print('i : {}, : image name : {} done'.format(i, data_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist', data=self.datalist)