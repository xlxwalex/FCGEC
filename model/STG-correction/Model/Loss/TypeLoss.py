import torch
from torch import nn
from argparse import Namespace

class TypeLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True):
        super(TypeLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.average = size_average

    def forward(self, logits : tuple, gts : torch.Tensor) -> torch.Tensor:
        logits_iwo, logits_iwc, logits_sc, logits_ill, logits_cm, logits_cr, logits_am = logits
        gts_iwo, gts_ip, gts_sc, gts_ill, gts_cm, gts_cr, gts_um = gts.T
        iwo_loss = self.criterion(logits_iwo, gts_iwo)
        iwc_loss  = self.criterion(logits_iwc,  gts_ip)
        sc_loss  = self.criterion(logits_sc,  gts_sc)
        ill_loss = self.criterion(logits_ill, gts_ill)
        cm_loss  = self.criterion(logits_cm,  gts_cm)
        cr_loss  = self.criterion(logits_cr,  gts_cr)
        am_loss  = self.criterion(logits_am,  gts_um)
        combine_loss = iwo_loss + iwc_loss + sc_loss + ill_loss + cm_loss + cr_loss + am_loss
        if self.average:
            combine_loss /= 7
        return combine_loss
