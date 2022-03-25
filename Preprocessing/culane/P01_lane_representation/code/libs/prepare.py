from datasets.dataset_culane import *
from visualizes.visualize import *

def prepare_dataloader(cfg, dict_DB):
    dataset = Dataset_Train(cfg=cfg)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=False)
    dict_DB['dataloader'] = dataloader
    return dict_DB

def prepare_visualization(cfg, dict_DB):
    dict_DB['visualize'] = Visualize(cfg)
    return dict_DB

