from Trainer.Trainer import Trainer
import torch
from tqdm import tqdm
import numpy as np
import time
import os
from utils import padding, attention_mask, save_model, SwitchMetric, SwitchSearch
from utils.metric import SwitchMetric_Spec
from torch.utils.data import DataLoader
from Model.Loss import SwitchLoss
from Model.Loss import TypeLoss

class SwitchTrainer(Trainer):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(SwitchTrainer, self).__init__(args, model, criterion, optimizer, device, checkp, scheduler)
        self.eval_inform = {'loss': [], 'token_acc': [], 'sentence_acc': []}
        self.train_loss = []
        self.eval_loss = []
        self.metric = SwitchMetric(args, mode = 'all')
        self.spec_metric = SwitchMetric_Spec(args, mode='all')
        self.loss_fn = SwitchLoss(args, device, criterion)
        self.decoder = SwitchSearch(args, args.sw_mode)

    # Train Model
    def train(self, Trainset: DataLoader, Validset: DataLoader):
        self.optimizer.zero_grad()
        self.step, pred_label, truth_label, met_masks = 0, [], [], []
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(Trainset):
                st_time = time.time()
                self.model.train()
                tokens, labels = batch_data
                padded = padding(tokens, self.args.padding_size, self.args.padding_val)
                attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
                padded = torch.from_numpy(padded).to(self.device)
                pad_label = padding(labels, self.args.padding_size, self.args.ignore_val)
                labels_tr = torch.from_numpy(pad_label).to(self.device)
                pointer_logits, pointer_masks = self.model(padded, attn_mask, need_mask = True)
                loss = self.loss_fn(pointer_logits, labels_tr, pointer_masks)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                self.optimizer.zero_grad()
                truths = padding(labels, self.args.padding_size, self.args.padding_val)
                met_masks.extend(attn_mask.detach().cpu().numpy())
                truth_label.extend(truths)
                preds = self.decoder(pointer_logits.detach().cpu(), attn_mask.detach().cpu().numpy())
                pred_label.extend(preds)
                if (self.step + 1) % self.args.print_step == 0:
                    met_acc = self.metric(truth_label, pred_label, met_masks)
                    token_acc, sent_acc = met_acc['token'], met_acc['sentence']
                    pred_label, truth_label, met_masks = [], [], []
                    print("step: %s, ave loss = %f, token_acc  = %f, sentence_acc = %f, speed: %f steps/s" %
                          (self.step + 1, self.train_loss[-1], token_acc, sent_acc, 1 / (time.time() - st_time)))
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    loss, token_acc, sent_acc = self.valid(Validset)
                    print(
                        "Final validation result: step: %d, ave loss: %f, ave token_acc: %f, sentence_acc = %f, speed: %f s/total" %
                        (self.step, loss, token_acc, sent_acc, 1 / (time.time() - eval_time)))
                    if self.best < token_acc:
                        save_model(os.path.join(self.checkpoint, 'checkpoint.pt'), self._generate_checkp())
                        print("Model Reached Best Performance, Save To Check_points")
                        self.best = token_acc
                self.step += 1
            self.epoch += 1


    # Valid Model
    def valid(self, Validset: DataLoader):
        self.model.eval()
        pred_label, truth_label, met_masks, eval_loss = [], [], [], []
        for step, batch_data in enumerate(Validset):
            tokens, labels = batch_data
            padded = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
            padded = torch.from_numpy(padded).to(self.device)
            pad_label = padding(labels, self.args.padding_size, self.args.ignore_val)
            labels_tr = torch.from_numpy(pad_label).to(self.device)
            with torch.no_grad():
                pointer_logits, pointer_masks = self.model(padded, attn_mask, need_mask = True)
            loss = self.loss_fn(pointer_logits, labels_tr, pointer_masks)
            eval_loss.append(loss.item())
            preds = np.argmax(pointer_logits.detach().cpu().numpy(), axis=1).astype('int32')
            truths = padding(labels, self.args.padding_size, self.args.padding_val)
            pred_label.extend(preds)
            met_masks.extend(attn_mask.detach().cpu().numpy())
            truth_label.extend(truths)
        met_acc = self.metric(truth_label, pred_label, met_masks)
        # Record
        self.eval_loss.append(np.mean(eval_loss))
        self.eval_inform['token_acc'].append(met_acc['token'])
        self.eval_inform['sentence_acc'].append(met_acc['sentence'])
        return self.eval_loss[-1], met_acc['token'], met_acc['sentence']

    # Test Model
    def test(self, Testset: DataLoader):
        self.model.eval()
        pred_logits, truth_label, met_masks, eval_loss, tokens_ls = None, [], [], [], []
        for step, batch_data in enumerate(tqdm(Testset, desc='Processing')):
            tokens, labels = batch_data
            tokens_ls.extend(tokens)
            padded = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
            padded = torch.from_numpy(padded).to(self.device)
            with torch.no_grad():
                pointer_logits = self.model(padded, attn_mask)
            truths = padding(labels, self.args.padding_size, self.args.padding_val)
            pred_logits = torch.cat((pred_logits, pointer_logits), dim=0) if pred_logits is not None else pointer_logits
            met_masks.extend(attn_mask.detach().cpu().numpy())
            truth_label.extend(truths)
        pred_label = self.decoder(pred_logits.detach().cpu(), met_masks)
        met_acc = self.metric(truth_label, pred_label, met_masks)
        spec_met_acc = self.spec_metric(tokens_ls, truth_label, pred_label, met_masks, out_put=True)
        return met_acc['token'], met_acc['sentence'], spec_met_acc

class SwitchTrainerTTI(Trainer):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(SwitchTrainerTTI, self).__init__(args, model, criterion, optimizer, device, checkp, scheduler)
        self.eval_inform = {'loss': [], 'token_acc': [], 'sentence_acc': []}
        self.train_loss = []
        self.eval_loss = []
        self.metric = SwitchMetric(args, mode = 'all')
        self.spec_metric = SwitchMetric_Spec(args, mode='all')
        self.loss_fn = SwitchLoss(args, device, criterion[0]) if criterion is not None else None
        self.tti_loss_fn = TypeLoss(args, device, criterion[1]) if criterion is not None else None
        self.decoder = SwitchSearch(args, args.sw_mode)

    # Train Model
    def train(self, Trainset: DataLoader, Validset: DataLoader):
        self.optimizer.zero_grad()
        self.step, pred_label, truth_label, met_masks = 0, [], [], []
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(Trainset):
                st_time = time.time()
                self.model.train()
                tokens, types, labels = batch_data
                padded = padding(tokens, self.args.padding_size, self.args.padding_val)
                attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
                padded = torch.from_numpy(padded).to(self.device)
                pad_label = padding(labels, self.args.padding_size, self.args.ignore_val)
                labels_tr = torch.from_numpy(pad_label).to(self.device)
                types_tr = torch.from_numpy(np.array(types)).to(self.device)
                pointer_logits, net, pointer_masks = self.model(padded, attn_mask, need_mask = True)
                loss_pointer = self.loss_fn(pointer_logits, labels_tr, pointer_masks)
                loss_tti = self.tti_loss_fn(net, types_tr)
                loss = loss_pointer + loss_tti
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                self.optimizer.zero_grad()
                truths = padding(labels, self.args.padding_size, self.args.padding_val)
                met_masks.extend(attn_mask.detach().cpu().numpy())
                truth_label.extend(truths)
                preds = self.decoder(pointer_logits.detach().cpu(), attn_mask.detach().cpu().numpy())
                pred_label.extend(preds)
                if (self.step + 1) % self.args.print_step == 0:
                    met_acc = self.metric(truth_label, pred_label, met_masks)
                    token_acc, sent_acc = met_acc['token'], met_acc['sentence']
                    pred_label, truth_label, met_masks = [], [], []
                    print("step: %s, ave loss = %f, token_acc  = %f, sentence_acc = %f, speed: %f steps/s" %
                          (self.step + 1, self.train_loss[-1], token_acc, sent_acc, 1 / (time.time() - st_time)))
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    loss, token_acc, sent_acc = self.valid(Validset)
                    # Evaluating Information
                    print(
                        "Final validation result: step: %d, ave loss: %f, ave token_acc: %f, sentence_acc = %f, speed: %f s/total" %
                        (self.step, loss, token_acc, sent_acc, 1 / (time.time() - eval_time)))
                    # Save Checkpoints For Best Model
                    if self.best < token_acc:
                        save_model(os.path.join(self.checkpoint, 'checkpoint.pt'), self._generate_checkp())
                        print("Model Reached Best Performance, Save To Check_points")
                        # Save Model To ./checkpoints
                        self.best = token_acc
                self.step += 1
            self.epoch += 1


    # Valid Model
    def valid(self, Validset: DataLoader):
        self.model.eval()
        pred_label, truth_label, met_masks, eval_loss = [], [], [], []
        for step, batch_data in enumerate(Validset):
            tokens, types, labels = batch_data
            padded = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
            padded = torch.from_numpy(padded).to(self.device)
            pad_label = padding(labels, self.args.padding_size, self.args.ignore_val)
            labels_tr = torch.from_numpy(pad_label).to(self.device)
            types_tr = torch.from_numpy(np.array(types)).to(self.device)
            with torch.no_grad():
                pointer_logits, net, pointer_masks = self.model(padded, attn_mask, need_mask = True)
            loss_pointer = self.loss_fn(pointer_logits, labels_tr, pointer_masks)
            loss_tti = self.tti_loss_fn(net, types_tr)
            loss = loss_pointer + loss_tti
            eval_loss.append(loss.item())
            preds = np.argmax(pointer_logits.detach().cpu().numpy(), axis=1).astype('int32')
            truths = padding(labels, self.args.padding_size, self.args.padding_val)
            pred_label.extend(preds)
            met_masks.extend(attn_mask.detach().cpu().numpy())
            truth_label.extend(truths)
        met_acc = self.metric(truth_label, pred_label, met_masks)
        # Record
        self.eval_loss.append(np.mean(eval_loss))
        self.eval_inform['token_acc'].append(met_acc['token'])
        self.eval_inform['sentence_acc'].append(met_acc['sentence'])
        return self.eval_loss[-1], met_acc['token'], met_acc['sentence']

    # Test Model
    def test(self, Testset: DataLoader):
        self.model.eval()
        pred_logits, truth_label, met_masks, eval_loss, tokens_ls = None, [], [], [], []
        for step, batch_data in enumerate(tqdm(Testset, desc='Processing')):
            tokens, types, labels = batch_data
            tokens_ls.extend(tokens)
            padded = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
            padded = torch.from_numpy(padded).to(self.device)
            with torch.no_grad():
                pointer_logits, net = self.model(padded, attn_mask)
            truths = padding(labels, self.args.padding_size, self.args.padding_val)
            pred_logits = torch.cat((pred_logits, pointer_logits), dim=0) if pred_logits is not None else pointer_logits
            met_masks.extend(attn_mask.detach().cpu().numpy())
            truth_label.extend(truths)
        pred_label = self.decoder(pred_logits.detach().cpu(), met_masks)
        met_acc = self.metric(truth_label, pred_label, met_masks)
        spec_met_acc = self.spec_metric(tokens_ls, truth_label, pred_label, met_masks, out_put=True)
        return met_acc['token'], met_acc['sentence'], spec_met_acc