from .contrastive import Contrastive, DirectCLR, ContrastivePlus
from .w_mse import WMSE
from .byol import BYOL, HSIC, BYOLPlus
from .barlow import BarlowTwins, BarlowTwinsPlus


METHOD_LIST = ["contrastive", "w_mse", "byol", "directclr","barlowtwins","hsic"]


def get_method(name, IL):
    assert name in METHOD_LIST
    if name == "contrastive":
        if IL:
            return ContrastivePlus
        else:
            return Contrastive
    elif name == "w_mse":
        return WMSE
    elif name == "byol":
        if IL:
            return BYOLPlus
        else:
            return BYOL
    elif name == "directclr":
        return DirectCLR
    elif name == "barlowtwins":
        if IL:
            return BarlowTwinsPlus
        else:
            return BarlowTwins
    elif name == "hsic":
        return HSIC
