import numpy as np
from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.eval_tusimple = dict_DB['eval_tusimple']
        self.visualize = dict_DB['visualize']

    def init_data(self):
        self.result = {'out': {'mul': []}, 'gt': {'mul': []}, 'name': []}
        self.datalist = []

    def run(self, model, mode='val'):
        self.init_data()

        with torch.no_grad():
            model.eval()

            for i, self.batch in enumerate(self.testloader):  # load batch data
                self.batch['img'] = self.batch['img'].cuda()
                self.batch['seg_label'] = self.batch['seg_label'].cuda()
                img_name = self.batch['img_name'][0]

                out = dict()
                model.forward_for_encoding(self.batch['img'])
                model.forward_for_squeeze()
                model.forward_for_lane_feat_extraction()
                out.update(model.forward_for_lane_component_prediction())
                out.update(self.forward_model.initialize_for_nms(out, model, self.cfg.max_iter, self.cfg.thresd_nms_iou, self.cfg.thresd_nms_iou_upper, thresd_score=self.cfg.thresd_score))
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

                if i % 50 == 1:
                    print('image {} ---> {} done!'.format(i, img_name))

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + mode + '/pickle/', file_name='datalist', data=self.datalist)

        # evaluation
        return self.evaluation(mode)

    def evaluation(self, mode):
        metric = dict()

        acc, fp, fn = self.eval_tusimple.measure_accuracy(mode, mode_h=True)

        metric['acc'] = acc
        metric['fn'] = fn
        metric['fp'] = fp

        return metric
