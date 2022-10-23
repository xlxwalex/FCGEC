import torch
from torch import nn
from argparse import Namespace

class GeneratorLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True):
        super(GeneratorLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.average = size_average
        self.max_gen = args.max_generate
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, mlm_logits : torch.Tensor, mlm_tgts : torch.Tensor) -> torch.Tensor:
        output_mlm = self.softmax(mlm_logits)
        loss_mlm = self.criterion(output_mlm, mlm_tgts)
        return loss_mlm
