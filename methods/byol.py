import functools
from itertools import chain
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from model import get_model, get_head
from .base import BaseMethod
from .norm_mse import norm_mse_loss
from .identifylay import IdentifyLay, weight_contrastive_loss


class BYOL(BaseMethod):
    """ implements BYOL loss https://arxiv.org/abs/2006.07733 """

    def __init__(self, cfg):
        """ init additional target and predictor networks """
        super().__init__(cfg)
        self.pred = nn.Sequential(
            nn.Linear(cfg.emb, cfg.head_size),
            nn.BatchNorm1d(cfg.head_size),
            nn.ReLU(),
            nn.Linear(cfg.head_size, cfg.emb),
        )
        self.model_t, _ = get_model(cfg.arch, cfg.dataset)
        # self.head_t = get_head(self.out_size, cfg)
        if cfg.eval_type == "256-middle" or cfg.eval_type == "256-last":
            self.first_head = nn.Sequential(
                nn.Linear(self.out_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
            self.last_head = nn.Sequential(
                nn.Linear(256, self.out_size),
                nn.BatchNorm1d(self.out_size),
                nn.ReLU()
            )
            self.head_t = nn.Sequential(self.first_head, self.last_head)
        else:
            self.head_t = get_head(self.out_size, cfg)
        for param in chain(self.model_t.parameters(), self.head_t.parameters()):
            param.requires_grad = False
        self.update_target(0)
        self.byol_tau = cfg.byol_tau
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss
    
    def sync(self):
        if dist.is_initialized():
            for param in chain(self.model_t.parameters(), self.head_t.parameters(), self.model.parameters(), self.head.parameters()):
                t = param.data.detach()
                dist.broadcast(t, 0)
                param.data.copy_(t)

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_t.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def forward(self, samples):
        z = [self.pred(self.head(self.model(x))) for x in samples]
        with torch.no_grad():
            zt = [self.head_t(self.model_t(x)) for x in samples]

        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                loss += self.loss_f(z[i], zt[j]) + self.loss_f(z[j], zt[i])
        loss /= self.num_pairs
        return loss

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)

class BYOLPlus(BYOL):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.good_struct = IdentifyLay(cfg)
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2
        self.loss_weight = functools.partial(weight_contrastive_loss, tau=cfg.temperature, norm=cfg.norm)
    
    def forward(self, samples):
        z = [self.pred(self.head(self.model(x))) for x in samples]
        with torch.no_grad():
            zt = [self.head_t(self.model_t(x)) for x in samples]
        x_orig = samples[0].cuda(non_blocking=True)
        z_orig = z[0]
        W_struct = self.good_struct.get_goodneighbor_orig(x_orig)
        W_orig = self.good_struct(z_orig)

        loss = 0
        loss_weight = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                loss += self.loss_f(z[i], zt[j]) + self.loss_f(z[j], zt[i])
                loss_weight += self.loss_weight(z[i], zt[j], W_orig) + self.loss_weight(z[j], zt[i], W_orig)
        loss /= self.num_pairs

        loss_norm = self.lambda2 * torch.norm((W_orig - W_struct), p='fro')

        total_loss = loss + self.lambda1 * loss_weight + loss_norm

        return total_loss, loss.item(), loss_weight.item(), loss_norm.item()