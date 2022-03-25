import argparse

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--dataset_dir', default=None, help='dataset dir')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    return cfg