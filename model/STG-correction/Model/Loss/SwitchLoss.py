import torch
from torch import nn
from argparse import Namespace
from utils import softmax_logits

class SwitchLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True, denomitor : float = 1e-8):
        super(SwitchLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.denomitor = denomitor
        self.average = size_average
        try:
            self.gamma = args.swloss_gamma
        except:
            self.gamma = 1e-2

    def forward(self, logits : torch.Tensor, gts : torch.Tensor, masks : torch.Tensor = None) -> torch.Tensor:
        label_loss = self.criterion(logits, gts)
        if masks is not None:
            #mask_logits = logits * masks
            mask_logits = softmax_logits(logits) * masks
            order_logits = torch.cat([torch.diag_embed(torch.diag(mask_logits[ins], -1), offset=-1).unsqueeze(0) for ins in range(mask_logits.shape[0])], dim = 0)
            irorder_logits = mask_logits - order_logits
            order_loss =  torch.sum(torch.exp(irorder_logits), dim = [1, 2]) / (torch.sum(torch.exp(order_logits), dim=[1, 2]) + self.denomitor)
            if self.average:
                order_loss = torch.mean(order_loss)
            else:
                order_loss = torch.sum(order_loss)
            combine_loss = label_loss + order_loss
        else:
            combine_loss = label_loss
        return combine_loss
