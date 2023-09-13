from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .base import BaseMethod

def weight_contrastive_loss(x0, x1, M, s, tau, norm):
    bsize = x0.shape[0]
    eye_mask = torch.eye(bsize).cuda() * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    s = s.view(-1,1)
    logits00 = torch.exp(s*torch.cdist(x0, x0)*(2*M[0] - 1) / tau - eye_mask).mean()
    logits11 = torch.exp(s*torch.cdist(x1, x1)*(2*M[1] - 1) / tau - eye_mask).mean()
    logits01 = torch.exp(s*torch.cdist(x0, x1)*(2*M[2] - 1) / tau).mean()
    logits10 = torch.exp(s*torch.cdist(x1, x0)*(2*M[3] - 1) / tau).mean()
    loss = torch.log(1 + (logits00 + logits01 + logits10 + logits11) / 4)
    return loss

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

class IdentifyLay(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tau = -cfg.tau
        self.yeta = cfg.yeta # number of neighbors
        self.mu = cfg.mu # threadshold of good neighbors
        self.pretrained = self.get_pretrained()
        self.pretrained.requires_grad_ = False
        self.discriminator = nn.Sequential(
            nn.Linear(2*cfg.emb, cfg.emb),
            nn.ReLU(),
            nn.Linear(cfg.emb, 1),
            nn.ReLU()
            )
        self.discriminator2 = nn.Sequential(
            nn.Linear(cfg.emb, cfg.emb),
            nn.ReLU(),
            nn.Linear(cfg.emb, 1),
            nn.ReLU()
            )
    
    def get_pretrained(self):
        model = resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Identity()
        return model.cuda()
    
    def get_goodneighbor(self, x):
        N = x.shape[0]
        s = torch.zeros(N)
        with torch.no_grad():
            z = F.normalize(self.pretrained(x),dim=1)
        pred = z @ z.T
        _, indices = pred.topk(self.yeta + 1, dim=1, largest=True)
        neighbors = {}
        for i, item in enumerate(indices[:,1:]):
            neighbors[i] = item.tolist()
        for i in range(N):
            m = 0
            for j in neighbors[i]:
                s_ij = sum([1 if i in neighbors[l] else 0 for l in neighbors[j]])
                if s_ij > self.mu:
                    m += 1
            s[i] = m / self.yeta
        return s.cuda()
    
    def get_M(self, x1, x2):
        with torch.no_grad():
            z1 = F.normalize(self.pretrained(x1),dim=1)
            z2 = F.normalize(self.pretrained(x2),dim=1)
        M0 = torch.exp(torch.cdist(z1,z1)/self.tau)
        M1 = torch.exp(torch.cdist(z2,z2)/self.tau)
        M2 = torch.exp(torch.cdist(z1,z2)/self.tau)
        M3 = torch.exp(torch.cdist(z2,z1)/self.tau)
        return [M0, M1, M2, M3]

    def forward(self, z1, z2, concat=False):
        N = z1.shape[0]
        if concat:
            input_tensor = torch.cat([torch.cat([z1[i].expand(N, z1.shape[1]),z1], dim=1) for i in range(N)])
            M0 = self.discriminator(input_tensor).view(N,N)
            input_tensor = torch.cat([torch.cat([z2[i].expand(N, z2.shape[1]),z2], dim=1) for i in range(N)])
            M1 = self.discriminator(input_tensor).view(N,N)
            input_tensor = torch.cat([torch.cat([z1[i].expand(N, z2.shape[1]),z1], dim=1) for i in range(N)])
            M2 = self.discriminator(input_tensor).view(N,N)
            input_tensor = torch.cat([torch.cat([z2[i].expand(N, z1.shape[1]),z2], dim=1) for i in range(N)])
            M3 = self.discriminator(input_tensor).view(N,N)
        else:
            input_tensor = torch.cat([z1[i].expand(N, z1.shape[1]) - z1 for i in range(N)])
            M0 = F.normalize(self.discriminator2(F.normalize(input_tensor, dim=0, p=1)).view(N,N), dim=1, p=1)
            input_tensor = torch.cat([z2[i].expand(N, z2.shape[1]) - z2 for i in range(N)])
            M1 = F.normalize(self.discriminator2(F.normalize(input_tensor, dim=0, p=1)).view(N,N), dim=1, p=1)
            input_tensor = torch.cat([z1[i].expand(N, z2.shape[1]) - z1 for i in range(N)])
            M2 = F.normalize(self.discriminator2(F.normalize(input_tensor, dim=0, p=1)).view(N,N), dim=1, p=1)
            input_tensor = torch.cat([z2[i].expand(N, z1.shape[1]) - z2 for i in range(N)])
            M3 = F.normalize(self.discriminator2(F.normalize(input_tensor, dim=0, p=1)).view(N,N), dim=1, p=1)
        return [M0, M1, M2, M3]

        
class SimCLR(BaseMethod):
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

        total_loss = loss + self.lambda1 * loss_weight + loss_norm / 4
        
        return total_loss, loss.item(), loss_weight.item(), loss_norm.item()