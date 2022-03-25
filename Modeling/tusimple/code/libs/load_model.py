import torch
from networks.model import Model
from networks.loss import *

def load_model_for_test(cfg, dict_DB):
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(cfg.dir['weight_paper'] + f'checkpoint_tusimple_res_{cfg.backbone}')
    else:
        if cfg.param_name == 'trained_last':
            checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_final')
        elif cfg.param_name == 'max':
            checkpoint = torch.load(cfg.dir['weight'] + f'checkpoint_max_acc_tusimple_res_{cfg.backbone}')
    model = Model(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['model'] = model
    return dict_DB

def load_model_for_train(cfg, dict_DB):
    model = Model(cfg=cfg)
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=cfg.gamma)

    if cfg.resume == False:
        checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.milestones,
                                                         gamma=cfg.gamma,
                                                         last_epoch=checkpoint['epoch'])
        dict_DB['epoch'] = checkpoint['epoch'] + 1
        dict_DB['val_result'] = checkpoint['val_result']

    loss_fn = Loss_Function(cfg)

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB