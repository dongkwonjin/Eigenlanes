import os

import numpy as np
from libs.utils import *

class Evaluation_CULane_Official(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def measure(self, test_mode, category, sampling_step):  # measure performance
        print('culane official metric evaluation start!')
        if sampling_step == 1:
            res = call_culane_eval_custom(data_dir=self.cfg.dir['dataset'] + 'img/',
                                          exp_name='',
                                          output_path=self.cfg.dir['out'] + test_mode + '/results/',
                                          category=category)
        else:
            res = call_culane_eval_custom_sampling(data_dir=self.cfg.dir['dataset'] + 'img/',
                                                   exp_name='',
                                                   output_path=self.cfg.dir['out'] + test_mode + '/results/')

        F, P, R = self.measure_Fscore(res)
        print('culane official metric evaluation done!')
        return F, P, R

    def measure_Fscore(self, res):
        TP, FP, FN = 0, 0, 0
        for k, v in res.items():
            val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
            val_tp, val_fp, val_fn = int(v['tp']), int(v['fp']), int(v['fn'])
            TP += val_tp
            FP += val_fp
            FN += val_fn
            print(k, val)

        if TP + FP != 0:
            P = TP * 1.0 / (TP + FP)
        else:
            P = 0
        if TP + FN != 0:
            R = TP * 1.0 / (TP + FN)
        else:
            R = 0
        if P + R != 0:
            F = 2 * P * R / (P + R)
        else:
            F = 0
        print(F)
        return F, P, R

# from "https://github.com/cfzd/Ultra-Fast-Lane-Detection"
def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

def call_culane_eval_custom_sampling(data_dir, exp_name, output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir = os.path.join(output_path, exp_name)

    w_lane = 30
    iou = 0.5  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list = output_path.replace('results', 'list') + 'datalist.txt'
    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    out = os.path.join(output_path, 'txt', 'out.txt')

    res_all = {}
    os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (data_dir, detect_dir, data_dir, list, w_lane, iou, im_w, im_h, frame, out))
    res_all['res_sampling'] = read_helper(out)

    return res_all

def call_culane_eval_custom(data_dir, exp_name, output_path, category):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir = os.path.join(output_path, exp_name)

    w_lane = 30
    iou = 0.5  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list0 = os.path.join(data_dir, 'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir, 'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir, 'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir, 'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir, 'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir, 'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir, 'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir, 'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir, 'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0 = os.path.join(output_path, 'txt', 'out0_normal.txt')
    out1 = os.path.join(output_path, 'txt', 'out1_crowd.txt')
    out2 = os.path.join(output_path, 'txt', 'out2_hlight.txt')
    out3 = os.path.join(output_path, 'txt', 'out3_shadow.txt')
    out4 = os.path.join(output_path, 'txt', 'out4_noline.txt')
    out5 = os.path.join(output_path, 'txt', 'out5_arrow.txt')
    out6 = os.path.join(output_path, 'txt', 'out6_curve.txt')
    out7 = os.path.join(output_path, 'txt', 'out7_cross.txt')
    out8 = os.path.join(output_path, 'txt', 'out8_night.txt')

    res_all = {}
    if category == 'test_all':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))

        res_all['res_normal'] = read_helper(out0)
        res_all['res_crowd'] = read_helper(out1)
        res_all['res_night'] = read_helper(out8)
        res_all['res_noline'] = read_helper(out4)
        res_all['res_shadow'] = read_helper(out3)
        res_all['res_arrow'] = read_helper(out5)
        res_all['res_hlight'] = read_helper(out2)
        res_all['res_curve'] = read_helper(out6)
        res_all['res_cross'] = read_helper(out7)
    elif category == 'test0_normal':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list0, w_lane, iou, im_w, im_h, frame, out0))
        res_all['res_noline'] = read_helper(out0)
    elif category == 'test1_crowd':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list1, w_lane, iou, im_w, im_h, frame, out1))
        res_all['res_crowd'] = read_helper(out1)
    elif category == 'test2_hlight':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list2, w_lane, iou, im_w, im_h, frame, out2))
        res_all['res_hlight'] = read_helper(out2)
    elif category == 'test3_shadow':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list3, w_lane, iou, im_w, im_h, frame, out3))
        res_all['res_shadow'] = read_helper(out3)
    elif category == 'test4_noline':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list4, w_lane, iou, im_w, im_h, frame, out4))
        res_all['res_noline'] = read_helper(out4)
    elif category == 'test5_arrow':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list5, w_lane, iou, im_w, im_h, frame, out5))
        res_all['res_arrow'] = read_helper(out5)
    elif category == 'test6_curve':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list6, w_lane, iou, im_w, im_h, frame, out6))
        res_all['res_curve'] = read_helper(out6)
    elif category == 'test7_cross':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list7, w_lane, iou, im_w, im_h, frame, out7))
        res_all['res_cross'] = read_helper(out7)
    elif category == 'test8_night':
        os.system('./evaluation/culane/evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (data_dir, detect_dir, data_dir, list8, w_lane, iou, im_w, im_h, frame, out8))
        res_all['res_night'] = read_helper(out8)

    return res_all