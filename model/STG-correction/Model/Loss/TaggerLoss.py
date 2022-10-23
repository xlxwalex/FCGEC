import torch
from torch import nn
from argparse import Namespace
from Model.Layer import CRF

class TaggerLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True):
        super(TaggerLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.tagger_criterion = criterion[0]
        self.insmod_criterion = criterion[1]
        self.average = size_average
        self.max_gen = args.max_generate

    def forward(self, tagger_logits : torch.Tensor, comb_logits : torch.Tensor,
                tagger_gts : torch.Tensor, comb_gts :torch.Tensor) -> torch.Tensor:
        tagger_logits = tagger_logits.permute(0, 2, 1)
        comb_logits = comb_logits.permute(0, 2, 1)
        tagger_loss = self.tagger_criterion(tagger_logits, tagger_gts)
        combine_loss = tagger_loss
        # if torch.max(ins_gts) > 0:
        insmod_loss = self.insmod_criterion(comb_logits, comb_gts)
        combine_loss += insmod_loss
        return combine_loss
