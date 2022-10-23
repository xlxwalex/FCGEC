from Trainer.Trainer import Trainer
import torch
from  tqdm import tqdm
import numpy as np
import time
import os
from utils import padding, attention_mask, save_model, clip_maxgenerate, TaggerMetricV2
from torch.utils.data import DataLoader
from Model.Loss import TaggerLoss, TypeLoss


class TaggerTrainer(Trainer):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(TaggerTrainer, self).__init__(args, model, criterion, optimizer, device, checkp, scheduler)
        # Loss Function
        self.loss_fn = TaggerLoss(args, device, criterion) if criterion is not None else None
        self.eval_inform = {'loss': [], 'token_tagger_acc': [], 'sentence_tagger_acc': [], 'sentence_insmod_acc' : [], 'token_insmod_acc' : []}
        self.metric = TaggerMetricV2(args, mode='all', level='high')
        self.train_loss = []
        self.eval_loss = []

    # Train Model
    def train(self, Trainset: DataLoader, Validset: DataLoader):
        self.optimizer.zero_grad()
        self.step, pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks = 0, [], [], [], [], []
        last_checkname = ''
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(Trainset):
                st_time = time.time()
                self.model.train()
                # Process Data
                tokens, tagger, comb= batch_data
                padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
                attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
                token_padded  = torch.from_numpy(padded_token).to(self.device)
                padded_label  = padding(tagger, self.args.padding_size, self.args.padding_val)
                tagger_label  = torch.from_numpy(padded_label).to(self.device)
                padded_comb   = padding(comb, self.args.padding_size, self.args.ignore_val)
                comb_label    = clip_maxgenerate(torch.from_numpy(padded_comb), self.args.max_generate).to(self.device)
                # Model Value
                tagger_logits, comb_logits = self.model(token_padded, attn_mask)
                loss = self.loss_fn(tagger_logits, comb_logits, tagger_label, comb_label)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                # Clear Gradient
                self.optimizer.zero_grad()
                # Preds
                tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
                comb_preds   = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
                pred_tagger.extend(tagger_preds)
                pred_comb.extend(comb_preds)
                gts_tagger   = padding(tagger, self.args.padding_size, self.args.padding_val)
                gts_comb     = padding(comb, self.args.padding_size, self.args.padding_val)
                truth_tagger.extend(gts_tagger)
                truth_comb.extend(gts_comb)
                met_masks.extend(attn_mask.detach().cpu().numpy())
                if (self.step + 1) % self.args.print_step == 0:
                    pack_gts = {'tagger' : truth_tagger, 'insmod' : truth_comb}
                    pack_preds = {'tagger' : pred_tagger,'insmod' : pred_comb}
                    met_ret = self.metric(pack_gts, pack_preds, met_masks)
                    token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
                    token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
                    token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
                    pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks = [], [], [], [], []
                    print("step: %s, ave loss = %f, token_acc[tagger = %.4f, insmod_t = %.4f], sentence_acc[tagger = %.4f, insmod_t = %.4f] speed: %f steps/s" %
                          (self.step + 1, self.train_loss[-1], token_tagger_acc, token_insmod_acc, sent_tagger_acc, sent_insmod_acc, 1 / (time.time() - st_time)))
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    loss, met_ret = self.valid(Validset)
                    token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
                    token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
                    token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
                    # Evaluating Information
                    print(
                        "Final validation result: step: %d, ave loss: %f, token_acc[tagger = %.4f, insmod = %.4f], sentence_acc[tagger = %.4f, insmod = %.4f] speed: %f s/total" %
                        (self.step, loss, token_tagger_acc, token_insmod_acc, sent_tagger_acc, sent_insmod_acc, 1 / (time.time() - eval_time)))
                    # Save Checkpoints For Best Model
                    if self.best < sent_tagger_acc:
                        cur_check_name = 'checkpoint.pt'
                        save_model(os.path.join(self.checkpoint, cur_check_name), self._generate_checkp())
                        print("Model Reached Best Performance, Save To Check_points")
                        # Save Model To ./checkpoints
                        self.best = sent_tagger_acc
                    # Process Step
                self.step += 1
            self.epoch += 1

    # Valid Model
    def valid(self, Validset: DataLoader):
        self.model.eval()
        pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks, eval_loss =  [], [], [], [], [], []
        for step, batch_data in enumerate(Validset):
            # Process Data
            tokens, tagger, comb = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_padded = torch.from_numpy(padded_token).to(self.device)
            padded_label = padding(tagger, self.args.padding_size, self.args.padding_val)
            tagger_label = torch.from_numpy(padded_label).to(self.device)
            padded_comb = padding(comb, self.args.padding_size, self.args.ignore_val)
            comb_label = clip_maxgenerate(torch.from_numpy(padded_comb), self.args.max_generate).to(self.device)
            # Model Value
            with torch.no_grad():
                tagger_logits, comb_logits = self.model(token_padded, attn_mask)
            loss = self.loss_fn(tagger_logits, comb_logits, tagger_label, comb_label)
            eval_loss.append(loss.item())
            # Preds
            tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
            comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
            pred_tagger.extend(tagger_preds)
            pred_comb.extend(comb_preds)
            gts_tagger = padding(tagger, self.args.padding_size, self.args.padding_val)
            gts_comb = padding(comb, self.args.padding_size, self.args.padding_val)
            truth_tagger.extend(gts_tagger)
            truth_comb.extend(gts_comb)
            met_masks.extend(attn_mask.detach().cpu().numpy())
        pack_gts = {'tagger': truth_tagger, 'insmod': truth_comb}
        pack_preds = {'tagger': pred_tagger, 'insmod': pred_comb}
        met_ret = self.metric(pack_gts, pack_preds, met_masks)
        token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
        token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
        token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
        # Record
        self.eval_loss.append(np.mean(eval_loss))
        self.eval_inform['token_tagger_acc'].append(token_tagger_acc)
        self.eval_inform['sentence_tagger_acc'].append(sent_tagger_acc)
        self.eval_inform['token_insmod_acc'].append(token_insmod_acc)
        self.eval_inform['sentence_insmod_acc'].append(sent_insmod_acc)
        return self.eval_loss[-1], met_ret

    # Test Model
    def test(self, Testset: DataLoader):
        self.model.eval()
        pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks =  [], [], [], [], []
        for step, batch_data in enumerate(tqdm(Testset, desc='Processing')):
            # Process Data
            # Process Data
            tokens, tagger, comb = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_padded = torch.from_numpy(padded_token).to(self.device)
            padded_label = padding(tagger, self.args.padding_size, self.args.padding_val)
            tagger_label = torch.from_numpy(padded_label).to(self.device)
            padded_comb = padding(comb, self.args.padding_size, self.args.ignore_val)
            comb_label = clip_maxgenerate(torch.from_numpy(padded_comb), self.args.max_generate).to(self.device)
            # Model Value
            with torch.no_grad():
                tagger_logits, comb_logits = self.model(token_padded, attn_mask)
            # Preds
            tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
            comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
            pred_tagger.extend(tagger_preds)
            pred_comb.extend(comb_preds)
            gts_tagger = padding(tagger, self.args.padding_size, self.args.padding_val)
            gts_comb = padding(comb, self.args.padding_size, self.args.padding_val)
            truth_tagger.extend(gts_tagger)
            truth_comb.extend(gts_comb)
            met_masks.extend(attn_mask.detach().cpu().numpy())
        pack_gts = {'tagger': truth_tagger, 'insmod': truth_comb}
        pack_preds = {'tagger': pred_tagger, 'insmod': pred_comb}
        met_ret = self.metric(pack_gts, pack_preds, met_masks)
        token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
        token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
        token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
        print("Final test result: token_acc[tagger = %.4f, insmod = %.4f], sentence_acc[tagger = %.4f, insmod = %.4f]" %
            (token_tagger_acc, token_insmod_acc, sent_tagger_acc, sent_insmod_acc))
        return token_tagger_acc, sent_tagger_acc


class TaggerTrainerTTI(Trainer):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(TaggerTrainerTTI, self).__init__(args, model, criterion, optimizer, device, checkp, scheduler)
        # Loss Function
        self.loss_fn = TaggerLoss(args, device, criterion[:2]) if criterion is not None else None
        self.tti_loss_fn = TypeLoss(args, device, criterion[2]) if criterion is not None else None
        self.eval_inform = {'loss': [], 'token_tagger_acc': [], 'sentence_tagger_acc': [], 'sentence_insmod_acc' : [], 'token_insmod_acc' : []}
        self.metric = TaggerMetricV2(args, mode='all', level='normal')
        self.train_loss = []
        self.eval_loss = []

    # Train Model
    def train(self, Trainset: DataLoader, Validset: DataLoader):
        self.optimizer.zero_grad()
        self.step, pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks = 0, [], [], [], [], []
        last_checkname = ''
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(Trainset):
                st_time = time.time()
                self.model.train()
                # Process Data
                tokens, tagger, comb, types = batch_data
                padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
                attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
                token_padded  = torch.from_numpy(padded_token).to(self.device)
                padded_label  = padding(tagger, self.args.padding_size, self.args.padding_val)
                tagger_label  = torch.from_numpy(padded_label).to(self.device)
                padded_comb   = padding(comb, self.args.padding_size, self.args.ignore_val)
                comb_label    = clip_maxgenerate(torch.from_numpy(padded_comb), self.args.max_generate).to(self.device)
                types_tr      = torch.from_numpy(np.array(types)).to(self.device)
                # Model Value
                tagger_logits, comb_logits, net = self.model(token_padded, attn_mask)
                loss_taginsmod = self.loss_fn(tagger_logits, comb_logits, tagger_label, comb_label)
                loss_tti = self.tti_loss_fn(net, types_tr)
                loss = loss_taginsmod + loss_tti
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                # Clear Gradient
                self.optimizer.zero_grad()
                # Preds
                tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
                comb_preds   = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
                pred_tagger.extend(tagger_preds)
                pred_comb.extend(comb_preds)
                gts_tagger   = padding(tagger, self.args.padding_size, self.args.padding_val)
                gts_comb     = padding(comb, self.args.padding_size, self.args.padding_val)
                truth_tagger.extend(gts_tagger)
                truth_comb.extend(gts_comb)
                met_masks.extend(attn_mask.detach().cpu().numpy())
                if (self.step + 1) % self.args.print_step == 0:
                    pack_gts = {'tagger' : truth_tagger, 'insmod' : truth_comb}
                    pack_preds = {'tagger' : pred_tagger,'insmod' : pred_comb}
                    met_ret = self.metric(pack_gts, pack_preds, met_masks)
                    token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
                    token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
                    token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
                    pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks = [], [], [], [], []
                    print("step: %s, ave loss = %f, token_acc[tagger = %.4f, insmod_t = %.4f], sentence_acc[tagger = %.4f, insmod_t = %.4f] speed: %f steps/s" %
                          (self.step + 1, self.train_loss[-1], token_tagger_acc, token_insmod_acc, sent_tagger_acc, sent_insmod_acc, 1 / (time.time() - st_time)))
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    loss, met_ret = self.valid(Validset)
                    token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
                    token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
                    token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
                    # Evaluating Information
                    print(
                        "Final validation result: step: %d, ave loss: %f, token_acc[tagger = %.4f, insmod = %.4f], sentence_acc[tagger = %.4f, insmod = %.4f] speed: %f s/total" %
                        (self.step, loss, token_tagger_acc, token_insmod_acc, sent_tagger_acc, sent_insmod_acc, 1 / (time.time() - eval_time)))
                    # Save Checkpoints For Best Model
                    if self.best < sent_tagger_acc:
                        cur_check_name = 'checkpoint.pt'
                        save_model(os.path.join(self.checkpoint, cur_check_name), self._generate_checkp())
                        print("Model Reached Best Performance, Save To Check_points")
                        # Save Model To ./checkpoints
                        self.best = sent_tagger_acc
                    # Process Step
                self.step += 1
            self.epoch += 1

    # Valid Model
    def valid(self, Validset: DataLoader):
        self.model.eval()
        pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks, eval_loss =  [], [], [], [], [], []
        for step, batch_data in enumerate(Validset):
            # Process Data
            tokens, tagger, comb, types = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_padded = torch.from_numpy(padded_token).to(self.device)
            padded_label = padding(tagger, self.args.padding_size, self.args.padding_val)
            tagger_label = torch.from_numpy(padded_label).to(self.device)
            padded_comb = padding(comb, self.args.padding_size, self.args.ignore_val)
            comb_label = clip_maxgenerate(torch.from_numpy(padded_comb), self.args.max_generate).to(self.device)
            types_tr = torch.from_numpy(np.array(types)).to(self.device)
            # Model Value
            with torch.no_grad():
                tagger_logits, comb_logits, net = self.model(token_padded, attn_mask)
            loss_taginsmod = self.loss_fn(tagger_logits, comb_logits, tagger_label, comb_label)
            loss_tti = self.tti_loss_fn(net, types_tr)
            loss = loss_taginsmod + loss_tti
            eval_loss.append(loss.item())
            # Preds
            tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
            comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
            pred_tagger.extend(tagger_preds)
            pred_comb.extend(comb_preds)
            gts_tagger = padding(tagger, self.args.padding_size, self.args.padding_val)
            gts_comb = padding(comb, self.args.padding_size, self.args.padding_val)
            truth_tagger.extend(gts_tagger)
            truth_comb.extend(gts_comb)
            met_masks.extend(attn_mask.detach().cpu().numpy())
        pack_gts = {'tagger': truth_tagger, 'insmod': truth_comb}
        pack_preds = {'tagger': pred_tagger, 'insmod': pred_comb}
        met_ret = self.metric(pack_gts, pack_preds, met_masks)
        token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
        token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
        token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
        # Record
        self.eval_loss.append(np.mean(eval_loss))
        self.eval_inform['token_tagger_acc'].append(token_tagger_acc)
        self.eval_inform['sentence_tagger_acc'].append(sent_tagger_acc)
        self.eval_inform['token_insmod_acc'].append(token_insmod_acc)
        self.eval_inform['sentence_insmod_acc'].append(sent_insmod_acc)
        return self.eval_loss[-1], met_ret

    # Test Model
    def test(self, Testset: DataLoader):
        self.model.eval()
        pred_tagger, pred_comb, truth_tagger, truth_comb, met_masks =  [], [], [], [], []
        for step, batch_data in enumerate(tqdm(Testset, desc='Processing')):
            # Process Data
            tokens, tagger, comb, types = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_padded = torch.from_numpy(padded_token).to(self.device)
            padded_label = padding(tagger, self.args.padding_size, self.args.padding_val)
            tagger_label = torch.from_numpy(padded_label).to(self.device)
            padded_comb = padding(comb, self.args.padding_size, self.args.ignore_val)
            comb_label = clip_maxgenerate(torch.from_numpy(padded_comb), self.args.max_generate).to(self.device)
            # Model Value
            with torch.no_grad():
                tagger_logits, comb_logits, net = self.model(token_padded, attn_mask)
            # Preds
            tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
            comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
            pred_tagger.extend(tagger_preds)
            pred_comb.extend(comb_preds)
            gts_tagger = padding(tagger, self.args.padding_size, self.args.padding_val)
            gts_comb = padding(comb, self.args.padding_size, self.args.padding_val)
            truth_tagger.extend(gts_tagger)
            truth_comb.extend(gts_comb)
            met_masks.extend(attn_mask.detach().cpu().numpy())
        pack_gts = {'tagger': truth_tagger, 'insmod': truth_comb}
        pack_preds = {'tagger': pred_tagger, 'insmod': pred_comb}
        met_ret = self.metric(pack_gts, pack_preds, met_masks)
        token_ret, sentence_ret = met_ret['token'], met_ret['sentence']
        token_tagger_acc, sent_tagger_acc = token_ret['tagger'], sentence_ret['tagger']
        token_insmod_acc, sent_insmod_acc = token_ret['comb'], sentence_ret['comb']
        print("Final test result: token_acc[tagger = %.4f, insmod = %.4f], sentence_acc[tagger = %.4f, insmod = %.4f]" %
            (token_tagger_acc, token_insmod_acc, sent_tagger_acc, sent_insmod_acc))
        return token_tagger_acc, sent_tagger_acc