import numpy as np
from torch.nn import functional as F
import torch

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_auc

import pdb


@TRAINER_REGISTRY.register()
class CxrVanilla(TrainerX):
    """Vanilla baseline."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.val_monitor = dict()
        self.best_result = 0

    def forward_backward(self, batch):
        # a batch is a dict?
        img, label = self.parse_batch_train(batch)
        output = self.model(img)  # Here, output is not through sigmoid()
        loss = self.weightedbce_loss(True, output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        img = batch['img']
        label = batch['label']
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def weightedbce_loss(self, use_gpu, output, label):
        weight = torch.zeros(label.size())
        num_total = label.shape[0]
        if use_gpu:
            weight = weight.cuda()
        is_pos = label.data == 1
        for i in range(label.shape[1]):
            is_pos = label.data[:, i] == 1
            num_neg = label.data[:, i] == 0
            num_neg = num_neg.sum()
            num_pos = label.data[:, i] == 1
            num_pos = num_pos.sum()
            weight[:, i][is_pos] = torch.true_divide(num_neg, num_total)
            weight[:, i][~is_pos] = torch.true_divide(num_pos, num_total)
            output = output.float()
            label = label.float()
            weight = weight.float()
        return F.binary_cross_entropy_with_logits(output, label, weight)


    def train(self):
        """Generic training loops."""
        self.es_patience = 10
        self.es_counter = 0

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            es_trigger = self.after_epoch()
            if es_trigger:
                print("Early Stop and Save Model")
                break
        self.after_train()


    def after_epoch(self, do_early_stopping=True):
        es_trigger = False
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            curr_average_auc = curr_result['average_auc']
            self.val_monitor['epoch '+str(self.epoch+1)] = curr_result
            is_best = curr_average_auc > self.best_result
            if is_best:
                self.best_result = curr_average_auc
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best.pth.tar"
                )
                self.es_counter = 0
            else:
                self.es_counter += 1

            if do_early_stopping:
                if self.es_counter >= self.es_patience:
                    es_trigger = True

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        return es_trigger
