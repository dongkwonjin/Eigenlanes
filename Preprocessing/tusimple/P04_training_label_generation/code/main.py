from options.config import Config
from options.args import *
from libs.prepare import *
from libs.utils import *
from libs.preprocess1 import Preprocessing as preprocessing1
from libs.preprocess2 import Preprocessing as preprocessing2

def run_1(cfg, dict_DB):
    preprocess = preprocessing1(cfg, dict_DB)
    preprocess.run()

def run_2(cfg, dict_DB):
    preprocess = preprocessing2(cfg, dict_DB)
    preprocess.run()


def main():
    # option
    cfg = Config()
    cfg = parse_args(cfg)

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)

    # run
    run_1(cfg, dict_DB)
    run_2(cfg, dict_DB)

if __name__ == '__main__':
    main()
