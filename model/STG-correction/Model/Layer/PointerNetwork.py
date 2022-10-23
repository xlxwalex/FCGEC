# Import Libs
import torch
import torch.nn as nn
from Model.Layer import AttetionScore
from Model.Layer import Linear
import argparse

class PointerNetwork(nn.Module):
    '''
    Pointer Network Module
    '''
    def __init__(self, args : argparse.Namespace, device : torch.device):
        super(PointerNetwork, self).__init__()
        self.args = args
        self.device = device
        # Attention Layer
        self._attn_score = AttetionScore(args, device)
        # Dense
        self._query_embedding = Linear(args.lm_hidden_size, args.lm_hidden_size)
        self._key_embedding   = Linear(args.lm_hidden_size, args.lm_hidden_size)

    def forward(self, inputs : torch.Tensor, masks : torch.Tensor = None, need_mask : bool = False):
        query  = self._query_embedding(inputs)
        key    = self._key_embedding(inputs)
        attn_ret = self._attn_score(query, key, masks, need_mask = need_mask)
        if need_mask:
            scores, mask = attn_ret
            return scores, mask
        else:
            scores = attn_ret
            return scores