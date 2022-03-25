from visualizes.visualize import *

def prepare_visualization(cfg, dict_DB):

    dict_DB['visualize'] = Visualize(cfg)
    return dict_DB

