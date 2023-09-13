from  functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMethod
from .identifylay import IdentifyLay, weight_contrastive_loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lambd = 1 / cfg.emb
        self.bn = nn.BatchNorm1d(cfg.emb, affine=False)

    def forward(self, samples):
        bs = len(samples[0])
        y1 = samples[0]
        y2 = samples[1]
        z1 = self.head(self.model(y1))
        z2 = self.head(self.model(y2))

        c = torch.matmul(self.bn(z1).T, self.bn(z2)) / bs
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class BarlowTwinsPlus(BarlowTwins):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.good_struct = IdentifyLay(cfg)
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2
        self.loss_weight = partial(weight_contrastive_loss, tau=cfg.temperature, norm=cfg.norm)
    
    def forward(self, samples):
        bs = len(samples[0])

        x_orig = samples[0]
        z_orig = self.head(self.model(x_orig))
        W_struct = self.good_struct.get_goodneighbor_orig(x_orig)
        W_orig = self.good_struct(z_orig)

        y1 = samples[1]
        y2 = samples[2]
        z1 = self.head(self.model(y1))
        z2 = self.head(self.model(y2))

        
        if torch.distributed.is_initialized():
            c = torch.matmul(self.bn(z1).T, self.bn(z2)) / (bs*torch.distributed.get_world_size())
            torch.distributed.all_reduce(c)
        else:
            c = torch.matmul(self.bn(z1).T, self.bn(z2)) / bs

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        loss_weight = self.loss_weight(z1, z2, W_orig)
        loss_norm = self.lambda2 * torch.norm((W_orig - W_struct), p='fro')
        total_loss = loss + self.lambda1 * loss_weight + loss_norm

        return total_loss, loss.item(), loss_weight.item(), loss_norm.item()
        