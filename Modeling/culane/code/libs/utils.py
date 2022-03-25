import os
try:
    import pickle5 as pickle
except:
    import pickle
import shutil

import numpy as np
import random
import torch

global global_seed

global_seed = 123
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def _init_fn(worker_id):

    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# convertor
def to_tensor(data):
    return torch.from_numpy(data).cuda()

def to_np(data):
    return data.cpu().numpy()

def to_np2(data):
    return data.detach().cpu().numpy()

def to_3D_np(data):
    return np.repeat(np.expand_dims(data, 2), 3, 2)

def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text),
        f.close()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# pickle
def save_pickle(dir_name, file_name, data):

    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(dir_name)
    with open(dir_name + file_name + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

def record_config(cfg, logfile):
    logger("*******Configuration*******\n", logfile)

    data = {k: getattr(cfg, k) for k in cfg.__dir__() if '__' not in k}
    for key, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, int):
                    logger("%s : %s %d\n" % (key, k, v), logfile)
                elif isinstance(v, str):
                    logger("%s : %s %s\n" % (key, k, v), logfile)
                elif isinstance(v, float):
                    logger("%s : %s %f\n" % (key, k, v), logfile)

        elif isinstance(value, int):
            logger("%s : %d\n" % (key, value), logfile)
        elif isinstance(value, str):
            logger("%s : %s\n" % (key, value), logfile)
        elif isinstance(value, float):
            logger("%s : %f\n" % (key, value), logfile)

    # copy config file
    if cfg.run_mode == 'train' and cfg.resume == True:
        os.system('cp %s %s' %('./options/config.py', os.path.join(cfg.dir['out'] + 'train/log/config.py')))
