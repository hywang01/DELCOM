import torch
from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.engine.trainer import SimpleNet

import pdb


@TRAINER_REGISTRY.register()
class CxrCrossGrad(TrainerX):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_f = cfg.TRAINER.CG.EPS_F
        self.eps_d = cfg.TRAINER.CG.EPS_D
        self.alpha_f = cfg.TRAINER.CG.ALPHA_F
        self.alpha_d = cfg.TRAINER.CG.ALPHA_D

    def build_model(self):
        cfg = self.cfg

        print("Building F")
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)

        print("Building D")
        self.D = SimpleNet(cfg, cfg.MODEL, self.num_source_domains)
        self.D.to(self.device)
        print("# params: {:,}".format(count_num_param(self.D)))
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        self.register_model("D", self.D, self.optim_D, self.sched_D)

    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)

        input.requires_grad = True

        # Compute domain perturbation
        loss_d = F.cross_entropy(self.D(input), domain)
        loss_d.backward()
        grad_d = torch.clamp(input.grad.data, min=-0.1, max=0.1)
        input_d = input.data + self.eps_f * grad_d

        # Compute label perturbation
        #pdb.set_trace()
        input.grad.data.zero_()
        loss_f = self.weightedbce_loss(True, self.F(input), label)
        loss_f.backward()
        grad_f = torch.clamp(input.grad.data, min=-0.1, max=0.1)
        input_f = input.data + self.eps_d * grad_f

        input = input.detach()

        # Update label net
        loss_f1 = self.weightedbce_loss(True, self.F(input), label)
        loss_f2 = self.weightedbce_loss(True, self.F(input_d), label)
        loss_f = (1 - self.alpha_f) * loss_f1 + self.alpha_f * loss_f2
        self.model_backward_and_update(loss_f, "F")

        # Update domain net
        loss_d1 = F.cross_entropy(self.D(input), domain)
        loss_d2 = F.cross_entropy(self.D(input_f), domain)
        loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
        self.model_backward_and_update(loss_d, "D")

        loss_summary = {"loss_f": loss_f.item(), "loss_d": loss_d.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.F(input)

    
    def weightedbce_loss(self, use_gpu, output, label):
        weight = torch.zeros(label.size())
        num_total = label.shape[0]
        if use_gpu:
            weight = weight.cuda()
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