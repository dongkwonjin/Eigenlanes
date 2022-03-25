import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

from libs.utils import *

class Dataset_Train(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(self.cfg.dir['pre4'] + 'datalist')

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.to_tensor = transforms.ToTensor()

    def get_image(self, idx, flip=0):
        out = dict()
        img = Image.open(self.cfg.dir['dataset'] + 'train_set/clips/{}.jpg'.format(self.datalist[idx])).convert('RGB')
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        img = self.transform(img)
        out['img_rgb'] = img
        out['img'] = self.normalize(img)
        return out

    def get_label_seg(self, label, sf):
        out = dict()
        label = cv2.GaussianBlur(label, self.cfg.kernel_8, cv2.BORDER_DEFAULT)
        label = cv2.resize(label, dsize=(self.cfg.width // sf, self.cfg.height // sf), interpolation=1)
        label[label != 0] = 1
        out['seg_label'] = label
        return out

    def get_label_org(self, idx, flip=0):
        out = dict()
        data = load_pickle(self.cfg.dir['pre0_train'] + self.datalist[idx])
        org_label = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
        org_label = np.ascontiguousarray(org_label)
        for j in range(len(data['lanes'])):
            x_coord = np.float32(data['lanes'][j]).reshape(-1, 1)
            y_coord = np.float32(data['h_samples']).reshape(-1, 1)
            check = (x_coord[:, 0] != -2)
            x_coord = x_coord[check]
            y_coord = y_coord[check]
            pts_org = np.concatenate((x_coord, y_coord), axis=1)
            if len(x_coord) == 0:
                continue
            pts = np.copy(pts_org)
            pts[:, 0] = pts_org[:, 0] / (self.cfg.org_width - 1) * (self.cfg.width - 1)
            pts[:, 1] = (pts_org[:, 1] - self.cfg.crop_size) / (self.cfg.org_height - self.cfg.crop_size - 1) * (self.cfg.height - 1)
            pts = np.int32(pts).reshape((-1, 1, 2))
            org_label = cv2.polylines(org_label, [pts], False, 1, 4)
        if flip == 1:
            org_label = cv2.flip(org_label, 1)  # horizontal flip
        out['org_label'] = np.float32(org_label != 0)
        return out

    def get_label_cls_reg(self, idx, flip):
        out = dict()

        data = load_pickle(self.cfg.dir['pre4'] + self.datalist[idx])[flip]

        org_px_coord = data['org_px_coord']
        org_c = data['org_c']
        height = data['height']
        iou = data['iou']
        iou_upper = data['iou_upper']

        error = data['error']
        lane_num = len(data['org_px_coord'])

        cls_prob = np.zeros(self.cfg.n_clusters, dtype=np.int64)
        reg_offset = np.zeros((self.cfg.n_clusters, self.cfg.top_m), dtype=np.float32)
        is_pos_reg = np.zeros(self.cfg.n_clusters, dtype=np.int64)
        cls_height = np.zeros(self.cfg.n_clusters, dtype=np.int64)
        c = np.zeros((lane_num, self.cfg.top_m), dtype=np.float32)

        px_coord = np.array([])
        for i in range(lane_num):
            # cls
            check_cls = (iou[i] > self.cfg.thresd_iou_for_cls) * (iou_upper[i] > self.cfg.thresd_iou_upper_for_cls)
            if np.sum(check_cls) == 0:
                idx_max = np.argmax(iou[i])
                pos_cls_idx = np.int32([idx_max])
            else:
                pos_cls_idx = check_cls.nonzero()[0]

            cls_prob[pos_cls_idx] = 1
            cls_height[pos_cls_idx] = np.argmin(np.abs(self.cfg.height_class - height[i]))

            check_reg = (iou[i] > self.cfg.thresd_iou_for_reg) * (iou_upper[i] > self.cfg.thresd_iou_upper_for_reg)
            if np.sum(check_reg) == 0:
                idx_max = np.argmax(iou[i])
                pos_reg_idx = np.int32([idx_max])
            else:
                pos_reg_idx = check_reg.nonzero()[0]

            # reg
            offset = error[i][pos_reg_idx]
            if len(offset.shape) == 1:
                offset = offset.reshape(1, -1)
            check1 = (np.min(offset, axis=1) >= self.cfg.thresd_min_offset)
            check2 = (np.max(offset, axis=1) <= self.cfg.thresd_max_offset)
            offset_check = check1 * check2
            if np.sum(offset_check) == 0:
                idx_max = np.argmax(iou[i])
                pos_reg_idx = np.int32([idx_max])
            else:
                pos_reg_idx = pos_reg_idx[offset_check == True]
            offset = error[i][pos_reg_idx]

            reg_offset[pos_reg_idx] = offset
            is_pos_reg[pos_reg_idx] = 1

            c[i] = org_c[i]
            if len(px_coord) == 0:
                px_coord = org_px_coord[i].reshape(1, -1)
            else:
                px_coord = np.concatenate((px_coord, org_px_coord[i].reshape(1, -1)))

        if len(data['org_px_coord']) == 0:
            exist_check = 0
        else:
            exist_check = 1

        out['prob'] = cls_prob
        out['offset'] = reg_offset
        out['is_pos_reg'] = is_pos_reg
        out['height_prob'] = cls_height
        out['exist_check'] = exist_check
        out['gt_c'] = c
        out['org_px_coord'] = px_coord

        # get buffer
        out_buffer = self.get_buffer_for_batch(out)
        out.pop('org_px_coord')
        out.pop('gt_c')
        out.update(out_buffer)

        return out

    def get_buffer_for_batch(self, data):
        out = dict()

        px_coord = data['org_px_coord']
        gt_c = data['gt_c']

        buffer_size = 20
        buffer_gt_c = np.zeros((buffer_size, self.cfg.top_m), dtype=np.float32)
        buffer_px_coord = np.zeros((buffer_size, self.cfg.node_num), dtype=np.float32)
        buffer_gtnum = np.zeros((1), dtype=np.int32)

        buffer_gtnum[0] = len(px_coord)

        buffer_gt_c[:buffer_gtnum[0]] = gt_c

        if px_coord.shape[0] != 0:
            buffer_px_coord[:px_coord.shape[0]] = px_coord

        out['org_px_coord'] = buffer_px_coord
        out['gt_c'] = buffer_gt_c
        out['gt_num'] = buffer_gtnum

        return out

    def __getitem__(self, idx):
        out = dict()
        flip = random.randint(0, 1)
        out['img_name'] = self.datalist[idx]
        out.update(self.get_image(idx, flip))
        out.update(self.get_label_org(idx, flip))
        out.update(self.get_label_seg(label=out['org_label'], sf=self.cfg.scale_factor[0]))
        out.update(self.get_label_cls_reg(idx, flip))
        return out

    def __len__(self):
        return len(self.datalist)

class Dataset_Test(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(self.cfg.dir['pre0_test'] + 'datalist')

        if cfg.sampling == True:
            sampling = np.arange(0, len(self.datalist), cfg.sampling_step)
            self.datalist = np.array(self.datalist)[sampling].tolist()

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx):
        out = dict()

        img = Image.open(self.cfg.dir['dataset'] + 'test_set/clips/{}.jpg'.format(self.datalist[idx])).convert('RGB')
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        img = self.transform(img)

        out['img_rgb'] = img
        out['img'] = self.normalize(img)

        return out

    def get_label_org(self, idx):
        out = dict()
        data = load_pickle(self.cfg.dir['pre0_test'] + self.datalist[idx])

        org_label = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
        org_label = np.ascontiguousarray(org_label)

        for j in range(len(data['lanes'])):
            x_coord = np.float32(data['lanes'][j]).reshape(-1, 1)
            y_coord = np.float32(data['h_samples']).reshape(-1, 1)
            check = (x_coord[:, 0] != -2)

            x_coord = x_coord[check]
            y_coord = y_coord[check]
            pts_org = np.concatenate((x_coord, y_coord), axis=1)

            if len(x_coord) == 0:
                continue

            pts = np.copy(pts_org)
            pts[:, 0] = pts_org[:, 0] / (self.cfg.org_width - 1) * (self.cfg.width - 1)
            pts[:, 1] = (pts_org[:, 1] - self.cfg.crop_size) / (self.cfg.org_height - self.cfg.crop_size - 1) * (self.cfg.height - 1)
            pts = np.int32(pts).reshape((-1, 1, 2))
            org_label = cv2.polylines(org_label, [pts], False, 1, 4)

        out['org_label'] = org_label

        return out

    def get_label_seg(self, label, sf):
        out = dict()
        label = cv2.GaussianBlur(label, self.cfg.kernel_8, cv2.BORDER_DEFAULT)
        label = cv2.resize(label, dsize=(self.cfg.width // sf, self.cfg.height // sf), interpolation=1)
        label[label != 0] = 1
        out['seg_label'] = label
        return out

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out.update(self.get_image(idx))
        out.update(self.get_label_org(idx))
        out.update(self.get_label_seg(label=out['org_label'], sf=self.cfg.scale_factor[0]))

        return out

    def __len__(self):
        return len(self.datalist)