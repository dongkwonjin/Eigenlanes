import time

import numpy as np
from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.save_prediction = dict_DB['save_prediction']
        self.eval_culane_laneatt = dict_DB['eval_culane_laneatt']
        self.eval_culane_official = dict_DB['eval_culane_official']
        self.visualize = dict_DB['visualize']

    def init_data(self):
        self.result = {'out': {'mul': []}, 'gt': {'mul': []}, 'name': []}
        self.datalist = []

    def run(self, model, mode='val'):
        self.init_data()

        with torch.no_grad():
            model.eval()
            for i, self.batch in enumerate(self.testloader):
                self.batch['img'] = self.batch['img'].cuda()
                self.batch['org_label'] = self.batch['org_label'].cuda()
                img_name = self.batch['img_name'][0]

                out = dict()
                model.forward_for_encoding(self.batch['img'])
                model.forward_for_squeeze()
                model.forward_for_lane_feat_extraction()
                out.update(model.forward_for_lane_component_prediction())
                out.update(self.forward_model.initialize_for_nms(out, model, self.cfg.max_iter, self.cfg.thresd_nms_iou))
                out.update(self.forward_model.run_for_nms(out, model))
                out.update(model.forward_for_matching(out['center_idx']))
                self.post_process.update(self.batch, out, mode)
                out.update(self.post_process.run(out))

                # visualize
                if self.cfg.disp_test_result == True:
                    if self.cfg.use_decoder == True:
                        out.update(model.forward_for_decoding())
                    self.visualize.display_for_test(batch=self.batch, out=out, batch_idx=i, mode=mode)

                # record output data
                self.result['out']['mwcs'] = to_tensor(out['mwcs'])
                self.result['out']['mwcs_reg'] = to_tensor(out['mwcs_reg'])
                self.result['out']['mwcs_h_idx'] = to_tensor(out['mwcs_height_idx'])
                self.result['out']['mwcs_vp_idx'] = to_tensor(out['mwcs_reg_vp_idx'])
                self.result['name'] = img_name
                self.datalist.append(img_name)

                if self.cfg.save_pickle == True:
                    dir_name, file_name = os.path.split(img_name)
                    save_pickle(dir_name=os.path.join(self.cfg.dir['out'] + '{}/pickle/{}/'.format(mode, dir_name)), file_name=file_name.replace('.jpg', ''), data=self.result)

                if i % 50 == 0:
                    print('image {} ---> {} done!'.format(i, img_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + mode + '/pickle/', file_name='datalist', data=self.datalist)

        # evaluation
        return self.evaluation(mode)

    def evaluation(self, mode):
        metric = dict()
        if self.cfg.do_eval_culane_laneatt == True:
            self.save_prediction.settings(key=['mwcs'], test_mode=mode, use_reg=True, use_height_cls=True)
            self.save_prediction.run()
            self.eval_culane_laneatt.settings(test_mode=mode)
            results = self.eval_culane_laneatt.measure()
            metric['precision_laneatt'] = results['Precision']
            metric['recall_laneatt'] = results['Recall']
            metric['fscore_laneatt'] = results['F1']

        if self.cfg.do_eval_culane_official == True:
            self.save_prediction.settings(key=['mwcs'], test_mode=mode, use_reg=True, use_height_cls=True)
            self.save_prediction.run()
            fscore, precision, recall = self.eval_culane_official.measure(test_mode=mode, category=self.cfg.category,
                                                                          sampling_step=self.testloader.dataset.sampling_step)
            metric['fscore_official'] = fscore
            metric['precision_official'] = precision
            metric['recall_official'] = recall

        return metric
