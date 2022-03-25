import os

from libs.utils import *
from options.config import Config

def main():

    # option
    cfg = Config()

    dir_namelist1 = os.listdir(cfg.dir['out'])
    for i in range(len(dir_namelist1)):
        dir_name1 = dir_namelist1[i]

        if 'display' in dir_name1:
            continue

        path1 = cfg.dir['out'] + dir_name1 + '/'

        dir_namelist2 = os.listdir(path1)
        for j in range(len(dir_namelist2)):
            dir_name2 = dir_namelist2[j]
            if 'pickle' in dir_name2:
                if 'datalist' in dir_name2:
                    datalist = load_pickle(path1 + dir_name2.replace('.pickle', ''))
                    datalist_t = list()
                    for k in range(len(datalist)):
                        f_name = datalist[k]

                        if '.jpg' in f_name:
                            f_name_t = f_name.replace('.jpg', '')
                            datalist_t.append(f_name_t)
                        else:
                            datalist_t.append(f_name)

                    save_pickle(path1, dir_name2.replace('.pickle', '_t'), datalist_t)
                continue

            if 'backup' in dir_name2:
                continue

            path2 = path1 + dir_name2 + '/'

            dir_namelist3 = os.listdir(path2)
            for k in range(len(dir_namelist3)):
                dir_name3 = dir_namelist3[k]

                path3 = path2 + dir_name3 + '/'

                file_namelist = os.listdir(path3)

                for l in range(len(file_namelist)):
                    file_name = file_namelist[l]
                    path4 = path3 + file_name
                    if '.jpg.jpg' in file_name:
                        src = path4
                        dst = path4.replace('.jpg', '')
                        os.rename(src, dst)
                    elif '.jpg.pickle' in file_name:
                        src = path4
                        dst = path4.replace('.jpg', '')
                        os.rename(src, dst)


if __name__ == '__main__':
    main()
