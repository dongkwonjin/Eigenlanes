import argparse

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='test_paper', help='run mode (train, test, test_paper)')
    parser.add_argument('--pre_dir', type=str, default='/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048/Work/CVPR2022/Roadlane/Project_github/test/Eigenlanes-main/preprocessed/culane/', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default='/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048/Dataset/CULane/', help='dataset dir')
    parser.add_argument('--paper_weight_dir', default='/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048/Work/CVPR2022/Roadlane/Project_github/test/Eigenlanes-main/pretrained/culane/', help='pretrained weights dir (paper)')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    if args.pre_dir is not None:
        cfg.dir['head_pre'] = args.pre_dir
        cfg.dir['pre0'] = cfg.dir['pre0'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre1'] = cfg.dir['pre1'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre2'] = cfg.dir['pre2'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre3'] = cfg.dir['pre3'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre4'] = cfg.dir['pre4'].replace('--preprocessed data path', args.pre_dir)
    cfg.dir['weight_paper'] = args.paper_weight_dir
    
    cfg.run_mode = args.run_mode
    if args.run_mode == 'test_paper':
        cfg.do_eval_culane_official = True
        cfg.sampling_step1 = 1

    return cfg
