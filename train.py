import os
import time
import signal
import subprocess
import numpy as np
import logging
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn

from cfg import get_cfg
from datasets import get_ds
from methods import get_method


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epoch * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.bs / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * 0.2
    optimizer.param_groups[1]['lr'] = lr * 0.0048
    
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None

def main_worker(gpu, cfg):
    cfg.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=cfg.dist_url,
        world_size=cfg.world_size, rank=cfg.rank)
    if cfg.rank == 0:
        logging.basicConfig(level=logging.INFO,filename=cfg.log)
        logger = logging.getLogger(__name__)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    per_device_batch_size = cfg.bs // cfg.world_size
    ds = get_ds(cfg.dataset)(per_device_batch_size, cfg, cfg.num_workers)
    model = get_method(cfg.method, cfg.IL)(cfg).cuda()
    if cfg.method == "byol":
        model.sync()
    if cfg.method != "barlowtwins":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu], find_unused_parameters=True)

    if cfg.optimizer == 'LARS':
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(parameters, lr=0, weight_decay=cfg.adam_l2,
                            weight_decay_filter=True,
                            lars_adaptation_filter=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    scheduler = get_scheduler(optimizer, cfg)
    eval_every = cfg.eval_every
    warm_up_epochs = (cfg.epoch // 2)
    lr_warmup = 0 if cfg.lr_warmup else warm_up_epochs

    for ep in range(cfg.epoch):
        if cfg.rank == 0:
            start = time.time()
        loss_ep = []
        loss_ep_contrastive = []
        loss_ep_weight = []
        loss_ep_norm = []
        iters = len(ds.train)
        for n_iter, (samples, _) in enumerate(ds.train):
            if lr_warmup < warm_up_epochs:
                lr_scale = (lr_warmup + 1) / warm_up_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1
            if cfg.optimizer == 'LARS':
                adjust_learning_rate(cfg, optimizer, ds.train, ep*len(ds.train)+n_iter)
            optimizer.zero_grad()
            if cfg.IL:
                loss, loss_contrastive, loss_weight, loss_norm = model(samples)
                loss_ep_contrastive.append(loss_contrastive)
                loss_ep_norm.append(loss_norm)
                loss_ep_weight.append(loss_weight)
            else:
                loss = model(samples[1:])
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            model.module.step(ep / cfg.epoch)
            if cfg.lr_step == "cos" and lr_warmup >= warm_up_epochs:
                scheduler.step(ep + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()

        if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
            eval_every = cfg.eval_every_drop

        if cfg.rank == 0:
            end = time.time()
            if (ep + 1) % eval_every == 0:
                acc_knn, acc = model.module.get_acc(ds.clf, ds.test)
                logger.info(str({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}))

            if (ep + 1) % 100 == 0:
                fname = f"data/{cfg.method}_{cfg.dataset}_{ep}_IL{cfg.IL}_{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pt"
                torch.save(model.module.state_dict(), fname)
            if cfg.IL:
                logger.info(str({"loss": np.mean(loss_ep_contrastive),
                "total loss": np.mean(loss_ep),
                "loss weight": np.mean(loss_ep_weight),
                "loss norm": np.mean(loss_ep_norm),
                "epoch time": end-start,
                "ep": ep}))
            else:
                logger.info(str({"loss": np.mean(loss_ep),
                "epoch time": end-start,
                "ep": ep}))




if __name__ == "__main__":
    cfg = get_cfg()
    main_worker(cfg)
