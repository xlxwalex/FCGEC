import torch
import torch.functional as F
from torch import nn
from Model.plm import PLM
from argparse import Namespace
from Model.Layer import Linear
from Model.Layer import CRF


class TaggerModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(TaggerModel, self).__init__()
        self.modelid = "tagger_baseline"
        self.args = args
        self.device = device
        self.max_token = args.max_generate + 1
        # Encoder
        self._encoder = PLM(args, device, use_encoder=True, pooler_output=False)
        # Solution A
        # | - Dense
        self._hidden2tag = Linear(args.lm_hidden_size, args.tagger_classes)
        self._hidden2t = Linear(args.lm_hidden_size, self.max_token)
        # | - Dropout
        self._lm_dropout = nn.Dropout(args.dropout)

    def forward(self, input : torch.Tensor, attention_mask : torch.Tensor = None):
        # Encoder
        encoded = self._encoder(input, attention_mask=attention_mask)
        encoded = self._lm_dropout(encoded)
        # Tagger
        tagger_logits = self._hidden2tag(encoded)
        # Generate Token
        t_logits = self._hidden2t(encoded)
        return tagger_logits, t_logits

class TaggerModelTTI(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(TaggerModelTTI, self).__init__()
        self.modelid = "tagger_baseline_tti"
        self.args = args
        self.device = device
        self.max_token = args.max_generate + 1
        # Encoder
        self._encoder = PLM(args, device, use_encoder=True, pooler_output=False, all_output=True)
        # Solution A
        # | - Dense
        self._hidden2tag = Linear(args.lm_hidden_size, args.tagger_classes)
        self._hidden2t = Linear(args.lm_hidden_size, self.max_token)
        # | - Dropout
        self._lm_dropout = nn.Dropout(args.dropout)
        # Linear
        self._fc_iwo = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_ip = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_sc = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_ill = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_cm = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_cr = Linear(args.lm_hidden_size, args.num_classes)
        self._fc_um = Linear(args.lm_hidden_size, args.num_classes)

    def forward(self, input : torch.Tensor, attention_mask : torch.Tensor = None):
        # Encoder
        pooler, encoded = self._encoder(input, attention_mask = attention_mask)
        encoded = self._lm_dropout(encoded)
        # Tagger
        tagger_logits = self._hidden2tag(encoded)
        # Generate Token
        t_logits = self._hidden2t(encoded)
        logit_iwo = self._fc_iwo(pooler)
        logit_iwc = self._fc_ip(pooler)
        logit_sc = self._fc_sc(pooler)
        logit_ill = self._fc_ill(pooler)
        logit_cm = self._fc_cm(pooler)
        logit_cr = self._fc_cr(pooler)
        logit_am = self._fc_um(pooler)
        type_logits = (logit_iwo, logit_iwc, logit_sc, logit_ill, logit_cm, logit_cr, logit_am)
        return tagger_logits, t_logits, type_logits