import math

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

from torch.nn import functional as F
from torch import Tensor

class Classifier(nn.Module):
    """
        d_x: node features
        d_e: edge features
    """

    def __init__(self, dx: int, de: int, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.linX = Linear(dx, 2, **kw)
        self.linE = Linear(de, 2, **kw)
        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor):
        ff_outputX = self.linX(X)
        ff_outputE = self.linE(E)

        return {
            'X': ff_outputX,
            'E': ff_outputE,
        }