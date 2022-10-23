from torch import nn
import torch
from argparse import Namespace

class LayerNorm(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, hidden_size : int = 768, eps : float =1e-6):
        super(LayerNorm, self).__init__()
        self.eps    = eps
        self.args   = args
        self.device = device
        self.gamma  = nn.Parameter(torch.ones(hidden_size))
        self.beta   = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        hidden_states =  self.gamma * (x-mean) / (std + self.eps)
        return hidden_states + self.beta