from sklearn.metrics import roc_auc_score
import numpy as np
import pdb


def compute_auc(gt, pred):
    # pred (torch.Tensor): model output [batch, num_classes]
    # gt (torch.LongTensor): ground truth [batch, num_classes]
    AUROCs = []
    gt_np = np.array(gt)
    pred_np = np.array(pred)
    for i in range(gt_np.shape[1]):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            #pdb.set_trace()
        except ValueError:
            AUROCs.append(0)
    return AUROCs