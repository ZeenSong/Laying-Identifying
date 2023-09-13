from torchvision.datasets import CIFAR10 as C10
from torch.utils.data import Dataset
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform, MultiSampleWithOrigin
from .base import BaseDataset
# import clip


def base_transform():
    return T.Compose(
        [T.Resize(32), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )


class CIFAR10(BaseDataset):
    def ds_train(self):
        # _, preprocess = clip.load("RN50")
        t = MultiSampleWithOrigin(
            aug_transform(32, base_transform, self.aug_cfg), base_transform(), n=self.aug_cfg.num_samples
        )
        return C10(root="./data", train=True, download=True, transform=t)

    def ds_clf(self):
        t = base_transform()
        return C10(root="./data", train=True, download=True, transform=t)

    def ds_test(self):
        t = base_transform()
        return C10(root="./data", train=False, download=True, transform=t)

class MyDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, (target, index)
