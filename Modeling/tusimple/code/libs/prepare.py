from datasets.dataset_tusimple import *
from visualizes.visualize import *
from tests.forward import *
from post_processes.post_process import *
from evaluation.eval_tusimple import LaneEval

from libs.utils import _init_fn
from libs.generator import *
from libs.load_model import *

def prepare_dataloader(cfg, dict_DB):
    # train dataloader
    dataset = Dataset_Train(cfg=cfg)
    trainloader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=cfg.batch_size['img'],
                                              shuffle=True,
                                              num_workers=cfg.num_workers,
                                              worker_init_fn=_init_fn)
    dict_DB['trainloader'] = trainloader

    # test dataloader
    dataset = Dataset_Test(cfg=cfg)
    testloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=False)
    dict_DB['testloader'] = testloader
    return dict_DB

def prepare_model(cfg, dict_DB):
    if 'test' in cfg.run_mode:
        dict_DB = load_model_for_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        dict_DB = load_model_for_train(cfg, dict_DB)

    dict_DB['forward_model'] = Forward_Model(cfg=cfg)
    return dict_DB

def prepare_visualization(cfg, dict_DB):
    dict_DB['visualize'] = Visualize_cv(cfg=cfg)
    return dict_DB

def prepare_evaluation(cfg, dict_DB):
    dict_DB['eval_tusimple'] = LaneEval(cfg=cfg)
    return dict_DB

def prepare_post_processing(cfg, dict_DB):
    dict_DB['post_process'] = Post_Processing(cfg=cfg)
    return dict_DB

def prepare_generator(cfg, dict_DB):
    dict_DB['generator'] = Label_Generator(cfg=cfg)
    return dict_DB

def prepare_training(cfg, dict_DB):
    logfile = cfg.dir['out'] + 'train/log/logfile.txt'
    mkdir(path=cfg.dir['out'] + 'train/log/')
    if cfg.run_mode == 'train' and cfg.resume == True:
        rmfile(path=logfile)
        val_result = dict()
        val_result['acc'] = 0
        dict_DB['val_result'] = val_result
        dict_DB['epoch'] = 0

        record_config(cfg, logfile)
    dict_DB['logfile'] = logfile
    return dict_DB


