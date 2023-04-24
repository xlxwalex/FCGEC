import torch
from argparse import Namespace
from Model import JointModel
from utils import padding, attention_mask, SwitchSearch, reconstruct_tagger_V2 as reconstruct_tagger, fillin_tokens
import os
import numpy as np

class ModelBucketV1(object):
    def __init__(self, args_demo: Namespace, device: torch.device, binary :bool = True, switch : bool=True, taggen : bool = True, checkpoints_name='checkpoint.pt'):
        self.binary_model   = None
        self.switch_model   = None
        self.tagger_model   = None
        self.generate_model = None
        self.device         = device
        self.args_demo      = args_demo
        # if binary:
        #     #self.binary_device = get_device(True, 0)
        #     self.binary_device = torch.device("cpu")
        #     self.binary_model = BERT(self.args_demo, self.binary_device)
        #     params = torch.load(BINARY_MODEL_PATH, map_location=self.binary_device)["model"]
        #     self.binary_model.load_state_dict(params)
        #     self.binary_model = self.binary_model.to(self.binary_device)
        #     self.binary_model.eval()
        #
        #     #self.type_device = get_device(True, 0)
        #     self.type_device = torch.device("cpu")
        #     params_types = torch.load(TYPE_MODEL_PATH, map_location=self.type_device)["model"]
        #     self.type_model = TypeBERT(self.args_type, self.type_device)
        #     self.type_model.load_state_dict(params_types)
        #     self.type_model = self.type_model.to(self.type_device)
        #     self.type_model.eval()

        joit_model_params = torch.load(os.path.join(self.args_demo.checkpoints, self.args_demo.checkp, checkpoints_name), map_location='cpu')["model"]
        joit_model = JointModel(args_demo, device)
        joit_model.load_state_dict(joit_model_params)
        joit_model = joit_model.to(device)
        joit_model.eval()

        if switch:
            self.switch_model = joit_model.switch
            self.sw_decoder = SwitchSearch(args=self.args_demo, mode=self.args_demo.sw_mode)

        if taggen:
            self.tagger_model = joit_model.tagger
            self.generate_model = joit_model.generator

    def binary_process(self, inputs : list):
        raise NotImplementedError # Not avaiblable in this version (Github)

    def switch_process(self, inputs):
        def _apply_switch_operator(wd_idxs: list, switch_ops: list) -> list:
            res = []
            for lidx in range(len(wd_idxs)):
                post_token = [101]
                switch_pred = switch_ops[lidx]
                sw_pidx = switch_pred[0]
                wd_idx = wd_idxs[lidx]
                while sw_pidx not in [0, -1]:
                    post_token.append(wd_idx[sw_pidx])
                    sw_pidx = switch_pred[sw_pidx]
                    if wd_idx[sw_pidx] == 102: switch_pred[sw_pidx] = 0
                res.append(post_token)
            return res

        if self.switch_model is None:
            raise Exception("Switch Model has not been initialized.")
        padded_token = padding(inputs, self.args_demo.padding_size, self.args_demo.padding_val)
        attn_mask = attention_mask(padded_token, self.args_demo.padding_val).to(self.device)
        token_padded = torch.from_numpy(padded_token).to(self.device)
        with torch.no_grad():
            pointer_logits = self.switch_model(token_padded, attn_mask)
        switch_preds = self.sw_decoder(pointer_logits.detach().cpu(), attn_mask.detach().cpu().numpy())
        return _apply_switch_operator(inputs, switch_preds), switch_preds

    def tagger_process(self, inputs):
        if self.tagger_model is None:
            raise Exception("Tagger Model has not been initialized.")
        padded_token = padding(inputs, self.args_demo.padding_size, self.args_demo.padding_val)
        attn_mask = attention_mask(padded_token, self.args_demo.padding_val).to(self.device)
        token_padded = torch.from_numpy(padded_token).to(self.device)
        with torch.no_grad():
            tagger_logits, comb_logits = self.tagger_model(token_padded, attn_mask)
        tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
        comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
        tag_construct = (tagger_preds, comb_preds)
        tag_tokens, mlm_tgt_masks, _ = reconstruct_tagger(padded_token, tag_construct)
        return (tagger_preds, comb_preds), (tag_tokens, mlm_tgt_masks)

    def generate_process(self, inputs : list):
        if self.generate_model is None:
            raise Exception("Generate Model has not been initialized.")
        tokens, masks = inputs
        padded = padding(tokens, self.args_demo.padding_size, self.args_demo.padding_val)
        gen_attn_mask = attention_mask(padded, self.args_demo.padding_val).to(self.device)
        padded = torch.from_numpy(padded).to(self.device)
        mask_padded = padding(masks, self.args_demo.padding_size, self.args_demo.padding_val)
        mask_padded = torch.from_numpy(mask_padded).to(self.device)
        with torch.no_grad():
            mlm_logits, tgt_mlm, _ = self.generate_model(padded, mask_padded, gen_attn_mask)
        token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
        return token_preds, fillin_tokens(tokens, masks, token_preds)