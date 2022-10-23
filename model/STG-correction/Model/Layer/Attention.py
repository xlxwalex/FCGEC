# Import Libs
import torch.nn as nn
import torch
import argparse
from utils import AttnMask, logits_mask

class AttetionScore(nn.Module):
    '''
    AttentionScore is used to calculate attention scores for Pointer Network
    '''
    def __init__(self, args : argparse.Namespace, device: torch.device = None):
        super(AttetionScore, self).__init__()
        self.device = device
        self.args = args
        try:
            self.scale_attn = args.scale_attn
        except:
            self.scale_attn = False

    def forward(self, query : torch.Tensor, key : torch.Tensor, mask : torch.Tensor = None, need_mask : bool = False):
        '''
        Calculate Attention Scores  For PointerNetwork
        :param query: Query tensor of shape `[batch_size, sequence_length, hidden_size]`.
        :param key: Key tensor of shape `[batch_size, sequence_length, hidden_size]`.
        :param mask: mask tensor of shape `[batch_size, sequence_length]`.
        :return: Tensor of shape `[batch_size, sequence_length, sequence_length]`.
        '''
        scores = torch.matmul(query, key.permute(0, 2, 1))
        if self.scale_attn:
            scores = scores * (1 / (query.shape[-1] ** (1/2)))
        if mask is not None:
            mask = AttnMask(scores, mask, diag_mask=False).to(self.device) if self.device else AttnMask(scores, mask, diag_mask=False).cuda()
            scores = logits_mask(scores, mask)
        if need_mask:
            return scores, mask
        else:
            return scores

