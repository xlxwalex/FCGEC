# Import Libs
import torch.nn as nn
import torch
import numpy as np
from torchcrf import CRF as crf
# Import Layers
from Model.Layer.Linear import Linear

class CRF(nn.Module):
    '''
    Condition Random Field Module
    '''
    def __init__(self, args, hidden_size : int, device : torch.device):
        super(CRF, self).__init__()
        self.modelname     = "crf"
        # INITIALIZE
        self.hidden_size = hidden_size
        self._crf        = crf(args.tagger_classes, batch_first=True).to(device)
        self._hidden2tag = Linear(self.hidden_size, args.tagger_classes)

    def forward(self, input : torch.Tensor, gt : torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:
        tag_encoded = self._hidden2tag(input)
        return - self._crf(tag_encoded, gt, reduction = 'mean', mask = mask)

    def decode(self, input : torch.Tensor) -> np.ndarray:
        tag_encoded = self._hidden2tag(input)
        return np.array(self._crf.decode(tag_encoded))