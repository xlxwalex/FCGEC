import torch
from torch import nn
from transformers import BertModel
from Model.Layer import *
import argparse

class PLM(nn.Module):
    def __init__(self, args : argparse.Namespace, device : torch.device, use_encoder : bool = False, pooler_output : bool = True, all_output : bool = False):
        super(PLM, self).__init__()
        self.modelid       = "bert_baseline"
        self.args          = args
        self.device        = device
        self.use_encoder   = use_encoder
        self.pooler_output = pooler_output
        self.all_output    = all_output
        self._bert         = BertModel.from_pretrained(args.lm_path, cache_dir='.cache/')
        # Finetune Or Freeze
        if args.finetune is not True:
            for param in self._bert.base_model.parameters():
                param.requires_grad = False
        # Modify BertModel - output_hidden_states
        self._bert_output = args.output_hidden_states
        self._bert.config.output_hidden_states = self._bert_output
        # Linear
        self._fc = Linear(args.lm_hidden_size, args.num_classes)

    def forward(self, inputs : torch.Tensor, attention_mask : torch.Tensor = None) -> tuple:
        encode_output = self._bert(inputs, attention_mask=attention_mask)
        # Pooler or Hidden
        if self.pooler_output:
            encoded = encode_output.pooler_output
        else:
            encoded = encode_output.last_hidden_state
        # Whether to apply dense
        if self.use_encoder is not True:
            output  = self._fc(encoded)
        else:
            if self.all_output:
                output = (encode_output.pooler_output, encode_output.last_hidden_state)
            else:
                output  = encoded
        return output

