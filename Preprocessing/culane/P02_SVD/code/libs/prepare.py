from datasets.dataset_culane import *
from visualizes.visualize import *

def prepare_dataloader(cfg, dict_DB):
    if 'train' in cfg.datalist:
        dataset = Dataset_Train(cfg=cfg)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=False)
    dict_DB['dataloader'] = dataloader

    return dict_DB

def prepare_visualization(cfg, dict_DB):
    dict_DB['visualize'] = Visualize(cfg)
    return dict_DB

