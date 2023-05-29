import torch
from torch import nn
from argparse import Namespace
from Model.Layer import Linear, LayerNorm, gelu, PointerNetwork
from Model.plm import UnifiedEncoder
from transformers import BertForMaskedLM, BertModel
from Model import SwitchModel, TaggerModel, GeneratorModel
from Model import SwitchModelEncoder, TaggerModelEncoder, GeneratorModelEncoder

class LMEncoder(nn.Module):
    def __init__(self, lm_path : str, finetune : bool = True, output_hidden_states : bool = True, dropout : float = 0.1):
        super(LMEncoder, self).__init__()
        self._lm = BertModel.from_pretrained(lm_path, cache_dir='.cache/')
        if finetune is not True:
            for param in self._lm.base_model.parameters():
                param.requires_grad = False
        self._lm_output = output_hidden_states
        self._lm.config.output_hidden_states = self._lm_output
        self._lm_dropout = nn.Dropout(dropout)

    def forward(self, inputs : torch.Tensor, attention_mask : torch.Tensor = None) -> tuple:
        encode_output = self._lm(inputs, attention_mask=attention_mask)
        pooler_output = encode_output.pooler_output
        last_hidden   = encode_output.last_hidden_state
        return pooler_output, last_hidden

class LMGenerator(nn.Module):
    def __init__(self, args : Namespace, lm_path : str, finetune : bool = True, device : torch.device = None):
        super(LMGenerator, self).__init__()
        self.args =args
        self._lmodel = BertForMaskedLM.from_pretrained(lm_path, cache_dir='.cache/')
        self.lmconfig = self._lmodel.config
        self.vocab_size = self.lmconfig.vocab_size
        if finetune is not True:
            for param in self._lm.base_model.parameters():
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

    def forward(self, inputs : torch.Tensor, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None):
        hidden_states = self._lmodel(inputs, attention_mask=attention_mask).logits
        tgt_mlm       = tgt_mlm.contiguous().view(-1)
        output_mlm    = hidden_states.contiguous().view(-1, self.vocab_size)
        output_mlm    = output_mlm[tgt_mlm > 0, :]
        tgt_mlm       = tgt_mlm[tgt_mlm > 0]
        denominator   = torch.tensor(output_mlm.size(0) + 1e-6)
        return output_mlm, tgt_mlm, denominator

class ClassificationLayer(nn.Module):
    def __init__(self, lm_hidden_size : int, num_classes : int):
        super(ClassificationLayer, self).__init__()
        # Binary Module
        self._fc     = Linear(lm_hidden_size, num_classes)
        # Type Module
        self._fc_iwo = Linear(lm_hidden_size, num_classes)
        self._fc_ip  = Linear(lm_hidden_size, num_classes)
        self._fc_sc  = Linear(lm_hidden_size, num_classes)
        self._fc_ill = Linear(lm_hidden_size, num_classes)
        self._fc_cm  = Linear(lm_hidden_size, num_classes)
        self._fc_cr  = Linear(lm_hidden_size, num_classes)
        self._fc_um  = Linear(lm_hidden_size, num_classes)

    def forward(self, pooler_output : torch.Tensor) -> tuple:
        bi_logits = self._fc(pooler_output)
        logit_iwo, logit_ip, logit_sc = self._fc_iwo(pooler_output).unsqueeze(0), self._fc_ip(pooler_output).unsqueeze(0), self._fc_sc(pooler_output).unsqueeze(0)
        logit_ill, logit_cm, logit_cr, logit_um = self._fc_ill(pooler_output).unsqueeze(0), self._fc_cm(pooler_output).unsqueeze(0), self._fc_cr(pooler_output).unsqueeze(0), self._fc_um(pooler_output).unsqueeze(0)
        type_logits = torch.cat((logit_iwo, logit_ip, logit_sc, logit_ill, logit_cm, logit_cr, logit_um), dim=0)
        return bi_logits, type_logits

class JointModelwithEncoder(nn.Module):
    def __init__(self,args : Namespace, device : torch.device):
        super(JointModelwithEncoder, self).__init__()
        self.max_token  = args.max_generate + 1
        self.encoder = UnifiedEncoder(args, device, use_encoder=True, pooler_output=False)
        self.switch = SwitchModelEncoder(args, self.encoder, device)
        self.tagger = TaggerModelEncoder(args, self.encoder, device)
        self.generator = GeneratorModelEncoder(args, self.encoder, device)

    def forward(self, inputs : tuple, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None):
        sw_inputs, tag_inputs, gen_inputs = inputs
        sw_attnmask, tag_attnmask, generate_attnmask = attention_mask
        switch_logits = self.switch(sw_inputs, sw_attnmask, need_mask=True)
        tagger_logits = self.tagger(tag_inputs, tag_attnmask)
        gen_logits = self.generator(gen_inputs, tgt_mlm, generate_attnmask)
        return switch_logits, tagger_logits, gen_logits

class JointModel(nn.Module):
    def __init__(self,args : Namespace, device : torch.device):
        super(JointModel, self).__init__()
        self.max_token  = args.max_generate + 1
        self.switch = SwitchModel(args, device)
        self.tagger = TaggerModel(args, device)
        self.generator = GeneratorModel(args, device)

    def forward(self, inputs : tuple, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None):
        sw_inputs, tag_inputs, gen_inputs = inputs
        sw_attnmask, tag_attnmask, generate_attnmask = attention_mask
        switch_logits = self.switch(sw_inputs, sw_attnmask, need_mask=True)
        tagger_logits = self.tagger(tag_inputs, tag_attnmask)
        gen_logits = self.generator(gen_inputs, tgt_mlm, generate_attnmask)
        return switch_logits, tagger_logits, gen_logits


class JointInferenceModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(JointInferenceModel, self).__init__()
        self.max_token = args.max_generate + 1
        # LM Encoder
        self.encoder = LMEncoder(args.lm_path, args.finetune, args.output_hidden_states)
        # Classification Layer (Binary, Types)
        self.cls_layer = ClassificationLayer(args.lm_hidden_size, args.num_classes)
        # Switch Layer
        self.pointer = PointerNetwork(args, device)
        # Tagger Layer + Generate Layer
        self.hidden2tag = Linear(args.lm_hidden_size, args.tagger_classes)
        self.hidden2ins = Linear(args.lm_hidden_size, self.max_token)
        self.hidden2mod = Linear(args.lm_hidden_size, self.max_token)
        self.generator  = LMGenerator(args, args.lm_path, args.finetune, device)

    def forward(self, inputs , stage : str, attention_mask : torch.Tensor = None, tgt_mlm : torch.Tensor = None, need_mask : bool = False):
        if stage == 'tag_before': # With binary & type
            orig_inputs, unify_attnmask = inputs, attention_mask
            pooler_ouptput, hidden_states = self.encoder(orig_inputs, attention_mask=unify_attnmask)
            bi_logits, cls_logits = self.cls_layer(pooler_ouptput)
            if need_mask:
                pointer_ret, masks = self.pointer(hidden_states, unify_attnmask, need_mask)
                return bi_logits, cls_logits, (pointer_ret, masks)
            else:
                pointer_ret = self.pointer(hidden_states, unify_attnmask, need_mask)
                return bi_logits, cls_logits, pointer_ret
        elif stage == 'tagger': # tagger stage
            tag_inputs, tag_attnmask = inputs, attention_mask
            _, tag_hidden_states = self.encoder(tag_inputs, attention_mask=tag_attnmask)
            # Tagger w (ins + mod logots)
            tagger_logits = self.hidden2tag(tag_hidden_states)
            ins_logits = self.hidden2ins(tag_hidden_states)
            mod_logits = self.hidden2mod(tag_hidden_states)
            tagger_logits = (tagger_logits, ins_logits, mod_logits)
            return tagger_logits
        elif stage == 'generator': # generator stage
            gen_inputs, generate_attnmask = inputs, attention_mask
            gen_logits = self.generator(gen_inputs, tgt_mlm, generate_attnmask)
            return gen_logits
        else: raise Exception('Model params `stage` error, please check.')