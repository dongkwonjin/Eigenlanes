import torch

from libs.prepare import *
from libs.save_model import *
from libs.utils import *

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']

        self.model = dict_DB['model']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']
        self.generator = dict_DB['generator']

        self.test_process = dict_DB['test_process']
        self.forward_model = dict_DB['forward_model']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']

        self.count = np.zeros(11, dtype=np.int32)

    def training(self):
        loss_t = dict()
        rmdir(path=self.cfg.dir['out'] + 'train/display_pos_edge/')
        rmdir(path=self.cfg.dir['out'] + 'train/display_neg_edge/')

        # train start
        self.model.train()
        print('train start')
        logger('train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):
            # load data
            for name in list(batch):
                if torch.is_tensor(batch[name]):
                    batch[name] = batch[name].cuda()

            # modeling
            out = dict()
            # ---------encoder--------- #
            self.model.forward_for_encoding(batch['img'])
            self.model.forward_for_squeeze()
            # --------SI module-------- #
            self.model.forward_for_lane_feat_extraction()
            out.update(self.model.forward_for_lane_component_prediction())
            # -----------NMS----------- #
            out.update(self.forward_model.initialize_for_nms(out, self.model, iteration=self.cfg.max_iter, thresd_nms_iou=self.cfg.thresd_nms_iou))
            out.update(self.forward_model.run_for_nms(out, self.model))
            # --------IC module-------- #
            out.update(self.generator.run(batch, out))
            out.update(self.model.forward_for_matching(out['node_idx']))
            # ---------decoder--------- #
            if self.cfg.use_decoder == True:
                out.update(self.model.forward_for_decoding())

            self.count += out['edge_count']

            # loss
            loss = self.loss_fn(out=out, gt=batch)
            # optimize
            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            for l_name in loss:
                if l_name not in loss_t.keys():
                    loss_t[l_name] = 0
                loss_t[l_name] += loss[l_name].item()


            if i % self.cfg.disp_step == 0:
                print('img iter {} ==> {}'.format(i, batch['img_name'][0]))
                self.visualize.display_for_train(batch, out, i)
                self.visualize.display_for_train_edge_score(out, i)
                if i % self.cfg.disp_step == 0:
                    logger("%d ==> " % i, self.logfile)
                    for l_name in loss:
                        logger("Loss_{} : {}, ".format(l_name, round(loss[l_name].item(), 4)), self.logfile)
                    logger("||| {}\n".format(batch['img_name'][0]), self.logfile)

            if i > 2000 // self.cfg.batch_size['img']:
                break

        # logger
        logger('\nAverage Loss : ', self.logfile)
        print('\nAverage Loss : ', end='')
        for l_name in loss_t:
            logger("{}, ".format(round(loss_t[l_name] / i, 6)), self.logfile)
        for l_name in loss_t:
            print("{}, ".format(round(loss_t[l_name] / i, 6)), end='')

        logger("\nlabel distribution {}\n".format(np.round((self.count / np.sum(self.count)), 3) * 100), self.logfile)
        print("\nlabel distribution {}\n".format(np.round((self.count / np.sum(self.count)), 3) * 100))

        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.model,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}

        save_model(checkpoint=self.ckpt,
                   param='checkpoint_final',
                   path=self.cfg.dir['weight'])

    def validation(self):
        # update testloader
        sampling_step = self.test_process.testloader.dataset.sampling_step
        if self.epoch >= self.cfg.epoch_eval_all and sampling_step != self.cfg.sampling_step2:
            self.test_process.testloader = update_test_dataloader(self.cfg, self.cfg.sampling_step2)

        metric = self.test_process.run(self.model, mode='val')

        logger("Epoch %3d ==> Validation result\n" % (self.ckpt['epoch']), self.logfile)
        if self.cfg.do_eval_culane_official == True:
            name_f = 'fscore_official'
            name_p = 'precision_official'
            name_r = 'recall_official'
            if name_f in metric.keys():
                logger("sampling step %d : mwcs %5f %5f %5f\n" % (sampling_step, metric[name_f], metric[name_p], metric[name_r]), self.logfile)
                print("sampling step %d : mwcs %5f %5f %5f\n" % (sampling_step, metric[name_f], metric[name_p], metric[name_r]))
            model_name = f'checkpoint_max_{name_f}_culane_res_{self.cfg.backbone}'
            if sampling_step == self.cfg.sampling_step2:
                self.val_result[name_f] = save_model_max(self.ckpt, self.cfg.dir['weight'],
                                                         self.val_result[name_f], metric[name_f],
                                                         logger, self.logfile, model_name)
        if self.cfg.do_eval_culane_laneatt == True:
            name_f = 'fscore_laneatt'
            name_p = 'precision_laneatt'
            name_r = 'recall_laneatt'
            if name_f in metric.keys():
                logger("sampling step %d : mwcs %5f %5f %5f\n" % (sampling_step, metric[name_f], metric[name_p], metric[name_r]), self.logfile)
                print("sampling step %d : mwcs %5f %5f %5f\n" % (sampling_step, metric[name_f], metric[name_p], metric[name_r]))
            model_name = f'checkpoint_max_{name_f}_culane_res_{self.cfg.backbone}'
            if sampling_step == self.cfg.sampling_step2:
                self.val_result[name_f] = save_model_max(self.ckpt, self.cfg.dir['weight'],
                                                         self.val_result[name_f], metric[name_f],
                                                         logger, self.logfile, model_name)

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch

            print('\nepoch {}\n'.format(epoch))
            logger('\nepoch {}\n'.format(epoch), self.logfile)

            self.training()
            # validation
            if epoch > self.cfg.epoch_eval:
                self.validation()
            elif epoch < 60 and epoch % 10 == 1:
                self.validation()
            elif 60 <= epoch and epoch < self.cfg.epoch_eval and epoch % 10 == 1:
                self.validation()

            self.scheduler.step(self.epoch)