import torch
import torch.functional as F
from torch import nn
from argparse import Namespace
from Model.Layer import Linear, LayerNorm, gelu
from transformers import BertForMaskedLM

class GeneratorModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(GeneratorModel, self).__init__()
        self.modelid = "generator_baseline"
        self.args = args
        self.device = device
        self._lmodel = BertForMaskedLM.from_pretrained(args.lm_path, cache_dir='.cache/')
        self.lmconfig = self._lmodel.config
        self.vocab_size = self.lmconfig.vocab_size
        # Finetune Or Freeze
        if args.finetune is not True:
            for param in self._bert.base_model.parameters():
                param.requires_grad = False
        # Mlm Linear Layers
        if self.args.factorized_embedding:
            self._mlm_fc_1 = Linear(args.lm_hidden_size, args.lm_emb_size)
            self._lnorm = LayerNorm(args, device, args.emb_size)
            self._mlm_fc_2 = Linear(args.lm_emb_size, self.vocab_size)
        else:
            self._mlm_fc_1 = Linear(args.lm_hidden_size, args.lm_hidden_size)
            self._lnorm = LayerNorm(args, device, args.lm_hidden_size)
            self._mlm_fc_2 = Linear(args.lm_hidden_size, self.vocab_size)
        # Activate Function
        self._act = gelu

    def forward(self, inputs : torch.Tensor, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None) -> tuple:
        # Encoded
        encoded = self._lmodel(inputs, attention_mask=attention_mask).logits
        # Mlm Linear Layer # 1
        # output_mlm = self._act(self._mlm_fc_1(encoded))
        # output_mlm = self._lnorm(output_mlm)
        # if self.factorized_embedding:
        #     output_mlm = output_mlm.contiguous().view(-1, self.args.lm_emb_size)
        # else:
        #     output_mlm = output_mlm.contiguous().view(-1, self.args.lm_hidden_size)
        # Extract Logits & Label
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = encoded.contiguous().view(-1, self.vocab_size)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        #output_mlm = self._mlm_fc_2(output_mlm)
        denominator = torch.tensor(output_mlm.size(0) + 1e-6).cuda()
        return output_mlm, tgt_mlm, denominator
