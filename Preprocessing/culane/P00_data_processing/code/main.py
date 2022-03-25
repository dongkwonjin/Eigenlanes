from options.config import Config
from options.args import *
from libs.preprocess import *

def run(cfg):
    preprocessor = Preprocessing(cfg)
    preprocessor.run()

def main():
    cfg = Config()
    cfg = parse_args(cfg)

    run(cfg)

if __name__ == '__main__':
    main()
