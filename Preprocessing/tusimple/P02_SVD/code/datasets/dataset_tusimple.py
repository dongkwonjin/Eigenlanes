import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.utils import *

class Dataset_Train(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(cfg.dir['pre1'] + 'datalist')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx, flip=0):
        img = Image.open(self.cfg.dir['dataset'] + 'train_set/clips/{}.jpg'.format(self.datalist[idx])).convert('RGB')
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        img = self.transform(img)

        return {'img': self.normalize(img),
                'img_rgb': img}

    def get_label(self, idx):
        data = load_pickle(self.cfg.dir['pre0'] + self.datalist[idx])

        seg_label = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
        seg_label = np.ascontiguousarray(seg_label)

        lane_pts = list()
        for j in range(len(data['lanes'])):

            x_coord = np.float32(data['lanes'][j]).reshape(-1, 1)
            y_coord = np.float32(data['h_samples']).reshape(-1, 1)
            check = (x_coord[:, 0] != -2)

            x_coord = x_coord[check]
            y_coord = y_coord[check]
            pts_org = np.concatenate((x_coord, y_coord), axis=1)

            if len(x_coord) == 0:
                continue

            lane_pts.append(pts_org)
            pts = np.copy(pts_org)
            pts[:, 0] = pts_org[:, 0] / (self.cfg.org_width - 1) * (self.cfg.width - 1)
            pts[:, 1] = (pts_org[:, 1] - self.cfg.crop_size) / (self.cfg.org_height - self.cfg.crop_size - 1) * (self.cfg.height - 1)
            pts = np.int32(pts).reshape((-1, 1, 2))
            seg_label = cv2.polylines(seg_label, [pts], False, 1, 4)

        return {'label': np.float32(seg_label),
                'lane_pts': lane_pts}

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out.update(self.get_image(idx))
        out.update(self.get_label(idx))

        return out

    def __len__(self):
        return len(self.datalist)
