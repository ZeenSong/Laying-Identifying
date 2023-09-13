import torch.nn as nn
from model import get_model, get_head
from eval.sgd import eval_sgd
from eval.knn import eval_knn
from eval.get_data import get_data


class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and head for training, evaluation function.
    """

    def __init__(self, cfg):
        super().__init__()
        self.model, self.out_size = get_model(cfg.arch, cfg.dataset)
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
            self.head = nn.Sequential(self.first_head, self.last_head)
        else:
            self.head = get_head(self.out_size, cfg)
        self.knn = cfg.knn
        self.num_pairs = cfg.num_samples * (cfg.num_samples - 1) // 2
        self.eval_head = cfg.eval_head
        self.eval_type = cfg.eval_type
        self.emb_size = cfg.emb

    def forward(self, samples):
        raise NotImplementedError

    def get_acc(self, ds_clf, ds_test):
        self.eval()
        if self.eval_head:
            model = lambda x: self.head(self.model(x))
            out_size = self.emb_size
        else:
            model, out_size = self.model, self.out_size

        x_train, y_train = get_data(model, ds_clf, out_size, "cuda")
        x_test, y_test = get_data(model, ds_test, out_size, "cuda")

        acc_knn = eval_knn(x_train, y_train, x_test, y_test, self.knn)
        acc_linear = eval_sgd(x_train, y_train, x_test, y_test)
        del x_train, y_train, x_test, y_test
        self.train()
        return acc_knn, acc_linear

    def step(self, progress):
        pass
