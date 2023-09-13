from functools import partial
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .base import BaseMethod
from .identifylay import IdentifyLay, weight_contrastive_loss

def contrastive_loss(x0, x1, tau, norm):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2
    
class Contrastive(BaseMethod):
    """ implements contrastive loss https://arxiv.org/abs/2002.05709 """

    def __init__(self, cfg):
        """ init additional BN used after head """
        super().__init__(cfg)
        self.bn_last = nn.BatchNorm1d(cfg.emb)
        self.loss_f = partial(contrastive_loss, tau=cfg.tau, norm=cfg.norm)

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.bn_last(self.head(torch.cat(h)))
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs : (i + 1) * bs]
                x1 = h[j * bs : (j + 1) * bs]
                loss += self.loss_f(x0, x1)
        loss_norm = 0

        loss /= self.num_pairs
        return loss
        
class DirectCLR(Contrastive):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.d0 = cfg.emb // 5

    def forward(self, samples):
        samples = samples[1:]
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True))[:,:self.d0] for x in samples]
        h = torch.cat(h)
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs : (i + 1) * bs]
                x1 = h[j * bs : (j + 1) * bs]
                loss += self.loss_f(x0, x1)
        loss /= self.num_pairs
        return loss
    

class ContrastivePlus(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.IL = IdentifyLay(cfg)
        self.bn_last = nn.BatchNorm1d(cfg.emb)
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2
        self.loss_f = partial(contrastive_loss, tau=cfg.tau, norm=cfg.norm)
        self.loss_weight = partial(weight_contrastive_loss, tau=cfg.temperature, norm=cfg.norm)
    
    def forward(self, samples):
        # get samples
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples[1:]]
        h = self.bn_last(self.head(torch.cat(h)))
        x_orig = samples[0].cuda(non_blocking=True)
        s = self.IL.get_goodneighbor(x_orig)
        
        loss = 0
        loss_weight = 0 
        M_pro = self.IL.get_M(samples[1],samples[2])
        x0 = h[:bs]
        x1 = h[bs:]
        M = self.IL(x0, x1)
        loss += self.loss_f(x0, x1)
        loss_weight += self.loss_weight(x0, x1, M, s)
        
        loss_norm = 0
        for i in range(len(M)):
            loss_norm += self.lambda2 * torch.norm((M[i] - M_pro[i]), p='fro')

        total_loss = loss + self.lambda1 * loss_weight + loss_norm
        
        return total_loss, loss.item(), loss_weight.item(), loss_norm.item()