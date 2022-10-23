from Trainer.Trainer import Trainer
import torch
from  tqdm import tqdm
import numpy as np
import time
import os
from utils import padding, attention_mask, save_model, GeneratorMetric, clip_maxgenerate
from torch.utils.data import DataLoader
from Model.Loss import GeneratorLoss

class GeneratorTrainer(Trainer):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(GeneratorTrainer, self).__init__(args, model, criterion, optimizer, device, checkp, scheduler)
        self.eval_inform = {'loss': [], 'token_generate_acc': [], 'sentence_generate_acc': []}
        self.metric = GeneratorMetric(args, mode='all')
        self.loss_fn = GeneratorLoss(args, device, criterion)
        self.train_loss = []
        self.eval_loss = []

    # Train Model
    def train(self, Trainset: DataLoader, Validset: DataLoader):
        self.optimizer.zero_grad()
        self.step, pred_mlm, truth_mlm, met_masks = 0, [], [], []
        last_checkname = ''
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(Trainset):
                st_time = time.time()
                self.model.train()
                # Process Data
                tokens, label = batch_data
                padded_token  = padding(tokens, self.args.padding_size, self.args.padding_val)
                attn_mask     = attention_mask(padded_token, self.args.padding_val).to(self.device)
                token_tensor  = torch.from_numpy(padded_token).to(self.device)
                padded_label  = padding(label, self.args.padding_size, self.args.padding_val)
                label_tensor  = torch.from_numpy(padded_label).to(self.device)
                # Model Value
                mlm_logits, tgt_mlm, _ = self.model(token_tensor, label_tensor, attn_mask)
                loss = self.loss_fn(mlm_logits, tgt_mlm)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                # Clear Gradient
                self.optimizer.zero_grad()
                # Preds
                token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
                token_truth = tgt_mlm.detach().cpu().numpy()
                pred_mlm.extend(token_preds)
                truth_mlm.extend(token_truth)
                met_masks.extend(label_tensor.detach().cpu().numpy())
                if (self.step + 1) % self.args.print_step == 0:
                    met_ret = self.metric(truth_mlm, pred_mlm, met_masks)
                    token_acc, sent_acc = met_ret['token'], met_ret['sentence']
                    pred_mlm, truth_mlm, met_masks = [], [], []
                    print("step: %s, ave loss = %f, token_acc  = %f, sentence_acc = %f, speed: %f steps/s" %
                          (self.step + 1, self.train_loss[-1], token_acc, sent_acc, 1 / (time.time() - st_time)))
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    loss, met_ret = self.valid(Validset)
                    token_acc, sent_acc = met_ret['token'], met_ret['sentence']
                    # Evaluating Information
                    print( "Final validation result: step: %d, ave loss: %f, ave token_acc: %f, sentence_acc = %f, speed: %f s/total" %
                        (self.step, loss, token_acc, sent_acc, 1 / (time.time() - eval_time)))
                    # Save Checkpoints For Best Model
                    if self.best < token_acc:
                        cur_check_name = 'checkpoint.pt'
                        save_model(os.path.join(self.checkpoint, cur_check_name), self._generate_checkp())
                        print("Model Reached Best Performance, Save To Check_points")
                        # Save Model To ./checkpoints
                        self.best = token_acc
                # Process Step
                self.step += 1
            self.epoch += 1

    # Valid Model
    def valid(self, Validset: DataLoader) -> tuple:
        self.model.eval()
        pred_mlm, truth_mlm, met_masks, eval_loss = [], [], [], []
        for step, batch_data in enumerate(Validset):
            # Process Data
            tokens, label = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_tensor = torch.from_numpy(padded_token).to(self.device)
            padded_label = padding(label, self.args.padding_size, self.args.padding_val)
            label_tensor = torch.from_numpy(padded_label).to(self.device)
            # Model Value
            with torch.no_grad():
                mlm_logits, tgt_mlm, _ = self.model(token_tensor, label_tensor, attn_mask)
            loss = self.loss_fn(mlm_logits, tgt_mlm)
            eval_loss.append(loss.item())
            # Preds
            token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
            token_truth = tgt_mlm.detach().cpu().numpy()
            pred_mlm.extend(token_preds)
            truth_mlm.extend(token_truth)
            met_masks.extend(label_tensor.detach().cpu().numpy())
        met_ret = self.metric(truth_mlm, pred_mlm, met_masks)
        # Record
        self.eval_loss.append(np.mean(eval_loss))
        self.eval_inform['token_generate_acc'].append(met_ret['token'])
        self.eval_inform['sentence_generate_acc'].append(met_ret['sentence'])
        return self.eval_loss[-1], met_ret

    # Test Model
    def test(self, Testset: DataLoader):
        self.model.eval()
        pred_mlm, truth_mlm, met_masks = [], [], []
        for step, batch_data in enumerate(Testset):
            # Process Data
            tokens, label = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_tensor = torch.from_numpy(padded_token).to(self.device)
            padded_label = padding(label, self.args.padding_size, self.args.padding_val)
            label_tensor = torch.from_numpy(padded_label).to(self.device)
            # Model Value
            with torch.no_grad():
                mlm_logits, tgt_mlm, _ = self.model(token_tensor, label_tensor, attn_mask)
            # Preds
            token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
            token_truth = tgt_mlm.detach().cpu().numpy()
            pred_mlm.extend(token_preds)
            truth_mlm.extend(token_truth)
            met_masks.extend(label_tensor.detach().cpu().numpy())
        met_ret = self.metric(truth_mlm, pred_mlm, met_masks)
        token_acc = met_ret['token']
        sentence_acc = met_ret['sentence']
        model_ret = {'prediction' : pred_mlm, 'label' : truth_mlm, 'mask' : met_masks}
        return token_acc, sentence_acc, model_ret