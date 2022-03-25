import torch.nn.functional as F

from libs.utils import *

class Forward_Model(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def initialize_for_nms(self, data, model, iteration, thresd_nms_iou, thresd_score=-1):
        out = dict(
            batch_size=data['prob'].shape[0],
            thresd_nms_iou=thresd_nms_iou,
            thresd_score=thresd_score,
            iteration=iteration
        )

        model.prob_map = torch.zeros((out['batch_size'], 1, self.cfg.n_clusters)).cuda()
        model.prob_map = data['prob']
        model.center_mask = torch.ones((out['batch_size'], 1, self.cfg.n_clusters)).cuda()
        model.visit_mask = torch.ones((out['batch_size'], 1, self.cfg.n_clusters)).cuda()
        model.batch_size = out['batch_size']
        model.thresd_nms_iou = out['thresd_nms_iou']
        model.thresd_score = out['thresd_score']

        return out

    def run_for_nms(self, data, model):
        iteration = data['iteration']
        batch_size = data['batch_size']
        prob = model.prob_map.clone()

        out = dict()
        out['center_idx'] = list()
        out['cluster_idx'] = list()

        for i in range(batch_size):
            for j in range(iteration):
                prob_updated = prob[i] * model.visit_mask[i]
                max_score, max_idx = model.selection_and_removal(prob_updated, batch_idx=i)

                if max_score < model.thresd_score:
                    break

            out['center_idx'].append((model.center_mask[i] == 0).nonzero()[:, 1])
            out['cluster_idx'].append((model.visit_mask[i] == 0).nonzero()[:, 1])

        out['center_idx'] = torch.cat(out['center_idx']).view(-1, iteration)
        return out
