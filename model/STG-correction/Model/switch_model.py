import torch
from torch import nn
from Model.plm import PLM
from argparse import Namespace
from Model.Layer import PointerNetwork, Linear

class SwitchModelEncoder(nn.Module):
    def __init__(self, args : Namespace, encoder, device : torch.device):
        super(SwitchModelEncoder, self).__init__()
        self.modelid = "switch_baseline_shared"
        self.args = args
        self.device = device
        # Encoder
        self._encoder = encoder
        # Pointer Network
        self._pointer = PointerNetwork(args, device)
        # Dropout
        self._lm_dropout = nn.Dropout(args.dropout)

    def forward(self, input: torch.Tensor, attention_mask : torch.Tensor = None, need_mask : bool = False):
        # Encoder
        encoded = self._encoder(input, attention_mask = attention_mask)
        encoded = self._lm_dropout(encoded)
        # Pointer Network
        pointer_ret = self._pointer(encoded, attention_mask, need_mask)
        if need_mask:
            pointer_logits, masks = pointer_ret
            return pointer_logits, masks
        else:
            pointer_logits = pointer_ret
            return pointer_logits

class SwitchModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(SwitchModel, self).__init__()
        self.modelid = "switch_baseline"
        self.args = args
        self.device = device
        # Encoder
        self._encoder = PLM(args, device, use_encoder=True, pooler_output=False)
        # Pointer Network
        self._pointer = PointerNetwork(args, device)
        # Dropout
        self._lm_dropout = nn.Dropout(args.dropout)

    def forward(self, input : torch.Tensor, attention_mask : torch.Tensor = None, need_mask : bool = False):
        # Encoder
        encoded = self._encoder(input, attention_mask = attention_mask)
        encoded = self._lm_dropout(encoded)
        # Pointer Network
        pointer_ret = self._pointer(encoded, attention_mask, need_mask)
        if need_mask:
            pointer_logits, masks = pointer_ret
            return pointer_logits, masks
        else:
            pointer_logits = pointer_ret
            return pointer_logits

class SwitchModelTTI(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(SwitchModelTTI, self).__init__()
        self.modelid = "switch_baseline_TTI"
        self.args = args
        self.device = device
        # Encoder
        self._encoder = PLM(args, device, use_encoder=True, pooler_output=False, all_output=True)
        # Pointer Network
        self._pointer = PointerNetwork(args, device)
        # Dropout
        self._lm_dropout = nn.Dropout(args.dropout)
        # Linear
        self._fc_iwo = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_ip = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_sc = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_ill = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_cm = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_cr = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_um = Linear(args.lm_hidden_size, args.num_classes)

    def forward(self, input : torch.Tensor, attention_mask : torch.Tensor = None, need_mask : bool = False):
        # Encoder
        pooler, encoded = self._encoder(input, attention_mask = attention_mask)
        encoded = self._lm_dropout(encoded)
        # Pointer Network
        pointer_ret = self._pointer(encoded, attention_mask, need_mask)
        logit_iwo = self._fc_iwo(pooler)
        logit_ip = self._fc_ip(pooler)
        logit_sc = self._fc_sc(pooler)
        logit_ill = self._fc_ill(pooler)
        logit_cm = self._fc_cm(pooler)
        logit_cr = self._fc_cr(pooler)
        logit_um = self._fc_um(pooler)
        type_logits = (logit_iwo, logit_ip, logit_sc, logit_ill, logit_cm, logit_cr, logit_um)
        if need_mask:
            pointer_logits, masks = pointer_ret
            return pointer_logits, type_logits, masks
        else:
            pointer_logits = pointer_ret
            return pointer_logits, type_logits
