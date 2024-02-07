import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import confusion_matrix
from ..metrics import compute_auc

from .build import EVALUATOR_REGISTRY

import pdb


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class CxrClassification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch, num_classesd]

        # Because mo from self.model is not through sigmoid()
        mo = torch.sigmoid(mo)

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(mo.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        auc_list = compute_auc(self._y_true, self._y_pred) # TODO
        # pdb.set_trace()
        class_names = self.cfg.DATASET.FINDING_NAMES
        for i in range(len(class_names)):
            results[class_names[i]] = auc_list[i]
        results['average_auc'] = np.array(auc_list).mean()

        print('=> per-class result')
        print('number of labels: ', len(self._y_true))
        print('number of predictions: ', len(self._y_pred))
        for i in range(len(class_names)):
            print(class_names[i] + ': ', '{:.4f}'.format(auc_list[i]))

        print('* average auc: {:.4f}'.format(results['average_auc']))

        return results
