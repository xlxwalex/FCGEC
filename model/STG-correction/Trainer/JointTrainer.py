# Import Libs
import torch
from  tqdm import tqdm
import numpy as np
import time
import os
from utils import padding, attention_mask, save_model, clip_maxgenerate, SwitchSearch, SwitchMetric, GeneratorMetric,  TaggerMetricV2
from utils import reconstruct_switch, reconstruct_tagger
from torch.utils.data import DataLoader
from Trainer.Trainer import Trainer
from Model.Loss import TaggerLoss, SwitchLoss, GeneratorLoss
from sklearn.metrics import  classification_report
from utils.metric import SwitchMetric_Spec


class JointTrainer(Trainer):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(JointTrainer, self).__init__(args, model, criterion, optimizer, device, checkp, scheduler)
        self.train_loss  = []
        self.eval_loss   = []
        self.all_acc     = []
        self.all_f1      = []
        # Loss Function
        if criterion is not None:
            self.sw_loss_fn     = SwitchLoss(args, device, criterion['sw'])
            self.tag_loss_fn = TaggerLoss(args, device, criterion['tag'])
            self.gen_loss_fn = GeneratorLoss(args, device, criterion['gen'])
        # Switch Decoder
        self.decoder     = SwitchSearch(args, args.sw_mode)
        # Metric Function
        self.sw_metric   = SwitchMetric(args, mode='all')
        self.tag_metric  = TaggerMetricV2(args , mode='all', level='normal')
        self.gen_metric  = GeneratorMetric(args, mode='all')
        self.spec_metric = SwitchMetric_Spec(args, mode='all')
        self.eval_inform = {'eval_inform' : [], 'metric_info' : [], 'loss_info' : []}
        self.tagger_weight = args.tagger_weight_post
        self.switch_weight = args.switch_weight
        self.switch_ref_weight = args.switch_ref_weight

    def collect_metric(self, swicths : tuple, tags : tuple, generates : tuple, desc : str = 'train'):
        ret = {'switch' : {}, 'tagger' : {}, 'generate' : {}}
        # Unpack Data
        switch_gts, switch_preds, switch_masks = swicths
        tagger_gts, tagger_preds, tagger_masks = tags
        gen_gts, gen_preds, gen_masks = generates
        # Switch
        switch_met = self.sw_metric(switch_gts, switch_preds, switch_masks)
        sw_acc_token, sw_acc_sentence = switch_met['token'], switch_met['sentence']
        ret['switch']['acc_token'] = sw_acc_token
        ret['switch']['acc_sent'] = sw_acc_sentence
        # Tagger
        tagger_met = self.tag_metric(tagger_gts, tagger_preds, tagger_masks)
        ret['tagger'] = tagger_met
        # Generate
        generate_met = self.gen_metric(gen_gts, gen_preds, gen_masks)
        ret['generate'] = generate_met
        train_acc_refer = [ret['switch']['acc_sent'] * self.switch_ref_weight, ret['tagger']['sentence']['tagger'], ret['tagger']['sentence']['comb'], ret['generate']['token']]
        # Process Desc for return
        if desc in ['train']:
            return sum(train_acc_refer) / len(train_acc_refer)
        else:
            return ret, sum(train_acc_refer) / len(train_acc_refer)

    # Train Model
    def train(self, Trainset: DataLoader, Validset: DataLoader):
        self.optimizer.zero_grad()
        self.step = 0
        switch_preds, switch_masks, switch_truths = [], [], []
        tag_preds, insmod_preds, tag_masks, tag_truths, insmod_truths = [], [], [], [], []
        mlm_preds, mlm_masks, mlm_truths = [], [], []
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(Trainset):
                st_time = time.time()
                self.model.train()
                # Process Data
                token_collection, tag_collection, label_collection = batch_data
                ori_tokens, tag_tokens, gen_tokens  = token_collection
                tag_label, insmod_label = tag_collection
                sw_label, mlm_label = label_collection
                # Token Padded
                padded_orig = padding(ori_tokens, self.args.padding_size, self.args.padding_val)
                padded_tags = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
                padded_gens = padding(gen_tokens, self.args.padding_size, self.args.padding_val)
                # Attention Masks
                ori_attn_mask = attention_mask(padded_orig, self.args.padding_val).to(self.device)
                tag_attn_mask = attention_mask(padded_tags, self.args.padding_val).to(self.device)
                gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
                # Tensor Trans
                ori_token_tensor = torch.from_numpy(padded_orig).to(self.device)
                tag_token_tensor = torch.from_numpy(padded_tags).to(self.device)
                gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
                # MLM Label
                padded_mlm     = padding(mlm_label, self.args.padding_size, self.args.padding_val)
                tgt_mlm_tensor = torch.from_numpy(padded_mlm).to(self.device)
                # Label Tensor
                pad_swlabel    = padding(sw_label, self.args.padding_size, self.args.ignore_val)
                swlabel_tensor = torch.from_numpy(pad_swlabel).to(self.device)
                padded_insmod  = padding(insmod_label, self.args.padding_size, self.args.ignore_val)
                insmod_label   = clip_maxgenerate(torch.from_numpy(padded_insmod), self.args.max_generate).to(self.device)
                padded_tagger  = padding(tag_label, self.args.padding_size, self.args.padding_val)
                tagger_label   = torch.from_numpy(padded_tagger).to(self.device)
                # Pack Data
                token_tensor   = (ori_token_tensor, tag_token_tensor, gen_token_tensor)
                attnmask_tensor= (ori_attn_mask, tag_attn_mask, gen_attn_mask)
                tagger_gts     = (tagger_label, insmod_label)
                # NetVal
                pointer_ret, tagger_logits, gen_logits = self.model(token_tensor, tgt_mlm_tensor, attnmask_tensor)
                # Unpack
                poi_ret, poi_mask = pointer_ret
                mlm_logits, mlm_tgts, _gen_logits = gen_logits
                tagger_logits, comb_logits = tagger_logits
                # Loss Process
                loss_switch = self.sw_loss_fn(poi_ret, swlabel_tensor, poi_mask)
                loss_tagger = self.tag_loss_fn(tagger_logits, comb_logits, tagger_label, insmod_label)
                loss_generator = self.gen_loss_fn(mlm_logits, mlm_tgts)
                loss = self.switch_weight * loss_switch + (self.tagger_weight if loss_tagger.item() < 1000 else 0.0005) * loss_tagger + loss_generator
                loss.backward()
                self.train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                # Clear Gradient
                self.optimizer.zero_grad()
                # Collect Result
                # | - Switch
                switch_pred = self.decoder(poi_ret.detach().cpu(), ori_attn_mask.detach().cpu().numpy())
                switch_preds.extend(switch_pred)
                switch_truths.extend(pad_swlabel)
                switch_masks.extend(ori_attn_mask.detach().cpu().numpy())
                # | - Tagger
                tag_pred = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
                insmod_pred = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
                tag_preds.extend(tag_pred)
                insmod_preds.extend(insmod_pred)
                tag_truths.extend(padded_tagger)
                insmod_truths.extend(padded_insmod)
                tag_masks.extend(tag_attn_mask.detach().cpu().numpy())
                # | - Generator
                mlm_pred = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
                mlm_preds.extend(mlm_pred)
                mlm_truths.extend(mlm_tgts.detach().cpu().numpy())
                mlm_masks.extend(padded_mlm)
                # Pack Result
                switch_res = (switch_truths, switch_preds, switch_masks)
                tagger_res = ({'tagger' : tag_truths, 'insmod' : insmod_truths}, {'tagger' : tag_preds, 'insmod' : insmod_preds}, tag_masks)
                gen_res    = (mlm_truths, mlm_preds, mlm_masks)
                if (self.step + 1) % self.args.print_step == 0:
                    metric_info = self.collect_metric(switch_res, tagger_res, gen_res)
                    print("step: %s, ave loss = %.4f, refer acc = %.4f, ..switch_loss = %.3f, tag_loss = %.3f, gen_loss = %.3f" %
                          (self.step + 1, loss.item(), metric_info, loss_switch, loss_tagger, loss_generator))
                    switch_preds, switch_masks, switch_truths = [], [], []
                    tag_preds, insmod_preds, tag_masks, tag_truths, insmod_truths = [], [], [], [], []
                    mlm_preds, mlm_masks, mlm_truths = [], [], []
                # Process Step
                self.step += 1
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    valid_loss, valid_metric, refer = self.valid(Validset)
                    eval_inform = eval_information(valid_loss, valid_metric)
                    print('Final validation result: step: %d, speed: %f s/total' % (self.step + 1, 1 / (time.time() - eval_time)))
                    print(eval_inform)
                    self.eval_inform['eval_inform'].append(eval_inform)
                    self.eval_inform['metric_info'].append(valid_metric)
                    self.eval_inform['loss_info'].append(valid_loss)
                    self.eval_loss.append(valid_loss)
                    # Save Checkpoints For Best Model
                    if self.best < refer:
                        save_model(os.path.join(self.checkpoint, 'checkpoint.pt'), self._generate_checkp())
                        print(">>>>>>> Model Reached Best Performance, Save To Check_points")
                        # Save Model To ./checkpoints
                        self.best = refer

    # Valid Model
    def valid(self, Validset: DataLoader) -> tuple:
        self.model.eval()
        switch_preds, switch_masks, switch_truths = [], [], []
        tag_preds, insmod_preds, tag_masks, tag_truths, insmod_truths = [], [], [], [], []
        mlm_preds, mlm_masks, mlm_truths = [], [], []
        eval_loss = {'overall' : [], 'switch' : [], 'tagger' : [], 'generate' : []}
        for step, batch_data in enumerate(Validset):
            # Process Data
            token_collection, tag_collection, label_collection = batch_data
            ori_tokens, tag_tokens, gen_tokens = token_collection
            tag_label, insmod_label = tag_collection
            sw_label, mlm_label = label_collection
            # Token Padded
            padded_orig = padding(ori_tokens, self.args.padding_size, self.args.padding_val)
            padded_tags = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
            padded_gens = padding(gen_tokens, self.args.padding_size, self.args.padding_val)
            # Attention Masks
            ori_attn_mask = attention_mask(padded_orig, self.args.padding_val).to(self.device)
            tag_attn_mask = attention_mask(padded_tags, self.args.padding_val).to(self.device)
            gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
            # Tensor Trans
            ori_token_tensor = torch.from_numpy(padded_orig).to(self.device)
            tag_token_tensor = torch.from_numpy(padded_tags).to(self.device)
            gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
            # MLM Label
            padded_mlm = padding(mlm_label, self.args.padding_size, self.args.padding_val)
            tgt_mlm_tensor = torch.from_numpy(padded_mlm).to(self.device)
            # Label Tensor
            pad_swlabel = padding(sw_label, self.args.padding_size, self.args.ignore_val)
            swlabel_tensor = torch.from_numpy(pad_swlabel).to(self.device)
            padded_insmod = padding(insmod_label, self.args.padding_size, self.args.ignore_val)
            insmod_label = clip_maxgenerate(torch.from_numpy(padded_insmod), self.args.max_generate).to(self.device)
            padded_tagger = padding(tag_label, self.args.padding_size, self.args.padding_val)
            tagger_label = torch.from_numpy(padded_tagger).to(self.device)
            # Pack Data
            token_tensor = (ori_token_tensor, tag_token_tensor, gen_token_tensor)
            attnmask_tensor = (ori_attn_mask, tag_attn_mask, gen_attn_mask)
            tagger_gts = (tagger_label, insmod_label)
            # NetVal
            with torch.no_grad():
                # NetVal
                pointer_ret, tagger_logits, gen_logits = self.model(token_tensor, tgt_mlm_tensor, attnmask_tensor)
                # Unpack
                poi_ret, poi_mask = pointer_ret
                mlm_logits, mlm_tgts, _gen_logits = gen_logits
                tagger_logits, comb_logits = tagger_logits
                # Loss Process
                loss_switch = self.sw_loss_fn(poi_ret, swlabel_tensor, poi_mask)
                loss_tagger = self.tag_loss_fn(tagger_logits, comb_logits, tagger_label, insmod_label)
                loss_generator = self.gen_loss_fn(mlm_logits, mlm_tgts)
                loss = loss_switch +0.01* loss_tagger + loss_generator
                eval_loss['overall'].append(loss.item())
                eval_loss['switch'].append(loss_switch.item())
                eval_loss['tagger'].append(0.01*loss_tagger.item())
                eval_loss['generate'].append(loss_generator.item())
            # Collect Result
            # | - Switch
            switch_pred = self.decoder(poi_ret.detach().cpu(), ori_attn_mask.detach().cpu().numpy())
            switch_preds.extend(switch_pred)
            switch_truths.extend(pad_swlabel)
            switch_masks.extend(ori_attn_mask.detach().cpu().numpy())
            # | - Tagger
            tag_pred = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
            insmod_pred = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
            tag_preds.extend(tag_pred)
            insmod_preds.extend(insmod_pred)
            tag_truths.extend(padded_tagger)
            insmod_truths.extend(padded_insmod)
            tag_masks.extend(tag_attn_mask.detach().cpu().numpy())
            # | - Generator
            mlm_pred = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
            mlm_preds.extend(mlm_pred)
            mlm_truths.extend(mlm_tgts.detach().cpu().numpy())
            mlm_masks.extend(padded_mlm)
        # Pack Result
        switch_res = (switch_truths, switch_preds, switch_masks)
        tagger_res = (
            {'tagger': tag_truths, 'insmod': insmod_truths}, {'tagger': tag_preds, 'insmod': insmod_preds}, tag_masks)
        gen_res = (mlm_truths, mlm_preds, mlm_masks)
        loss_info = gather_loss(eval_loss)
        metric_info, refer_acc = self.collect_metric(switch_res, tagger_res, gen_res, desc='valid')
        return loss_info, metric_info, refer_acc

    def test(self, Testset: DataLoader):
        self.model.eval()
        binary_preds, type_preds, switch_preds, switch_masks, binary_truths, type_truths, switch_truths = [], None, [], [], [], None, []
        tag_preds, ins_preds, mod_preds, tag_masks, tag_truths, ins_truths, mod_truths = [], [], [], [], [], [], []
        mlm_preds, mlm_masks, mlm_truths = [], [], []
        for step, batch_data in enumerate(tqdm(Testset, desc='Inferencing')):
            # Process Data
            token_collection, tag_collection, label_collection = batch_data
            ori_tokens, tag_tokens, gen_tokens = token_collection
            tag_label, ins_label, mod_label = tag_collection
            binary_label, type_label, sw_label, mlm_label = label_collection
            # Token Padded
            padded_orig = padding(ori_tokens, self.args.padding_size, self.args.padding_val)
            padded_tags = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
            # Attention Masks
            ori_attn_mask = attention_mask(padded_orig, self.args.padding_val).to(self.device)
            tag_attn_mask = attention_mask(padded_tags, self.args.padding_val).to(self.device)
            # Tensor Trans
            ori_token_tensor = torch.from_numpy(padded_orig).to(self.device)
            # MLM Label
            padded_mlm = padding(mlm_label, self.args.padding_size, self.args.padding_val)
            # tgt_mlm_tensor = torch.from_numpy(padded_mlm).to(self.device)
            # Label Tensor
            pad_swlabel = padding(sw_label, self.args.padding_size, self.args.ignore_val)
            padded_insert = padding(ins_label, self.args.padding_size, self.args.ignore_val)
            padded_modify = padding(mod_label, self.args.padding_size, self.args.ignore_val)
            padded_tagger = padding(tag_label, self.args.padding_size, self.args.padding_val)
            # Stage 1 : Binary + Type + Pointer
            with torch.no_grad():
                bi_logits, cls_logits, pointer_ret = self.model(ori_token_tensor, 'tag_before', ori_attn_mask, need_mask=True)
                poi_ret, poi_mask = pointer_ret
            # Processing 4 Tagger & Generator
            binary_pred = np.argmax(bi_logits.detach().cpu().numpy(), axis=1).astype('int32')
            binary_preds.extend(binary_pred)
            binary_truths.extend(binary_label)
            type_pred = np.argmax(cls_logits.detach().cpu().numpy(), axis=2).astype('int32').T
            type_preds = np.vstack((type_preds, type_pred)) if type_preds is not None else type_pred
            type_truths = np.vstack((type_truths, np.array(type_label))) if type_truths is not None else np.array(type_label)
            switch_pred = self.decoder(poi_ret.detach().cpu(), ori_attn_mask.detach().cpu().numpy())
            switch_preds.extend(switch_pred)
            switch_truths.extend(pad_swlabel)
            switch_masks.extend(ori_attn_mask.detach().cpu().numpy())
            switch_tokens = reconstruct_switch(padded_orig, switch_pred)
            tag_token_tensor = torch.from_numpy(switch_tokens).to(self.device)
            # Stage 2 : Tagger
            with torch.no_grad():
                tagger_logits = self.model(tag_token_tensor, 'tagger', tag_attn_mask)
                tag_logits, ins_logits, mod_logits = tagger_logits
            # | - Tagger
            tag_pred = np.argmax(tag_logits.detach().cpu().numpy(), axis=2).astype('int32')
            ins_pred = np.argmax(ins_logits.detach().cpu().numpy(), axis=2).astype('int32')
            mod_pred = np.argmax(mod_logits.detach().cpu().numpy(), axis=2).astype('int32')
            tag_preds.extend(tag_pred)
            ins_preds.extend(ins_pred)
            mod_preds.extend(mod_pred)
            tag_truths.extend(padded_tagger)
            ins_truths.extend(padded_insert)
            mod_truths.extend(padded_modify)
            tag_masks.extend(tag_attn_mask.detach().cpu().numpy())
            # Obtain generate tokens
            tag_construct = (tag_pred, ins_pred, mod_pred)
            tag_tokens, mlm_tgt_masks = reconstruct_tagger(switch_tokens, tag_construct)
            padded_gens = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
            gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
            gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
            padded_mlm_tgt_mask = padding(mlm_tgt_masks, self.args.padding_size, self.args.padding_val)
            tgt_mlm_tensor = torch.from_numpy(padded_mlm_tgt_mask).to(self.device)
            # Stage 3 : Generator
            with torch.no_grad():
                gen_logits = self.model(gen_token_tensor, 'generator', gen_attn_mask, tgt_mlm_tensor)
                mlm_logits, mlm_tgts, _gen_logits = gen_logits
            # | - Generator
            mlm_pred = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
            mlm_preds.extend(mlm_pred)
            mlm_truths.extend(mlm_tgts.detach().cpu().numpy())
            mlm_masks.extend(padded_mlm_tgt_mask)
        # Pack Result
        binary_res = (binary_preds, binary_truths)
        types_res = (type_preds, type_truths)
        switch_res = (switch_truths, switch_preds, switch_masks)
        tagger_res = ({'tagger': tag_truths, 'insert': ins_truths, 'modify': mod_truths}, {'tagger': tag_preds, 'insert': ins_preds, 'modify': mod_preds}, tag_masks)
        gen_res = (mlm_truths, mlm_preds, mlm_masks)
        metric_info, refer_acc = self.collect_metric(binary_res, types_res, switch_res, tagger_res, gen_res, desc='valid')
        print('>> Binary:\n' + classification_report(binary_truths, binary_preds) + '\n')
        print('>> Types:\n' + classification_report(type_truths, type_preds))
        tagger_res = ((tag_truths, ins_truths, mod_truths), (tag_preds, ins_preds, mod_preds), tag_masks)
        return metric_info, refer_acc, eval_information(None, metric_info), (binary_res, types_res, switch_res, tagger_res, gen_res)

    # Generate Checkpoints
    def _generate_checkp(self) -> dict:
        checkpoints = {
            'model': self.model.state_dict(),
            'optim': self.optimizer,
            'metric': self.eval_inform,
            'args': self.args,
            'epoch': self.epoch,
            'train_loss' : self.train_loss,
            'eval_loss' : self.eval_loss
        }
        return checkpoints

def gather_loss(loss_info : dict):
    gathered = {}
    for lokey in loss_info.keys(): gathered[lokey] = np.mean(loss_info[lokey])
    return gathered

def eval_information(loss_info : dict, metric_info : dict) -> str:
    if loss_info is not None:
        inform = ('>>' * 80 + '\n')
        inform += '> Loss Info: \n'
        for lokey in loss_info.keys(): inform += (lokey + 'loss = %.2f ' % loss_info[lokey])
        inform += '\n'
    else: inform = ''
    inform += '> Metric Info:\n'
    for metkey in metric_info.keys():
        inform += (metkey + ': ')
        sub_metric = metric_info[metkey]
        for skey in sub_metric.keys():
            if metkey != 'tagger': inform += (skey + ': %.4f ' % sub_metric[skey])
            else:
                inform += skey + ':['
                for sskey in sub_metric[skey]: inform += (sskey + ': %.4f ' % sub_metric[skey][sskey])
                inform+= '] '
        inform += '\n'
    inform += '>>' * 80
    return inform