import torch
from torch import nn
from argparse import Namespace
from utils import softmax_logits

class JointLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True, weights : dict = None, denomitor : float = 1e-8):
        super(JointLoss, self).__init__()
        self.args = args
        self.device = device
        self.cross_entropy = criterion['ce']
        self.ce_tagger = criterion['ce_tag']
        self.ce_operate = criterion['ce_ops']
        self.nll_loss = criterion['nll']
        self.average = size_average
        self.max_gen = args.max_generate
        self.weights = weights
        self.softmax = nn.LogSoftmax(dim=-1)
        self.denomitor = denomitor
        try:
            self.gamma = args.swloss_gamma
        except:
            self.gamma = 1e-2

    def pointer_loss(self, logits : torch.Tensor, gts : torch.Tensor, masks : torch.Tensor = None) -> torch.Tensor:
        label_loss = self.cross_entropy(logits, gts)
        if masks is not None:
            mask_logits = softmax_logits(logits) * masks
            order_logits = torch.cat([torch.diag_embed(torch.diag(mask_logits[ins], -1), offset=-1).unsqueeze(0) for ins in range(mask_logits.shape[0])], dim = 0)
            irorder_logits = mask_logits - order_logits
            order_loss =  torch.sum(torch.exp(irorder_logits), dim = [1, 2]) / (torch.sum(torch.exp(order_logits), dim=[1, 2]) + self.denomitor)
            if self.average:
                order_loss = torch.mean(order_loss)
            else:
                order_loss = torch.sum(order_loss)
            combine_loss = label_loss + order_loss
        else:
            combine_loss = label_loss
        return combine_loss

    def tagger_loss(self, tagger_preds : list, tagger_truth : list) -> torch.Tensor:
        tagger_logits, ins_logits, mod_logits = tagger_preds
        tagger_gts, ins_gts, mod_gts = tagger_truth
        tagger_logits = tagger_logits.permute(0, 2, 1)
        insert_logits = ins_logits.permute(0, 2, 1)
        modify_logits = mod_logits.permute(0, 2, 1)
        tagger_loss = self.ce_tagger(tagger_logits, tagger_gts)
        tag_combine_loss = tagger_loss
        if torch.max(ins_gts) > 0:
            insert_loss = self.ce_operate(insert_logits, ins_gts)
            tag_combine_loss += insert_loss
        if torch.max(mod_gts) > 0:
            modify_loss = self.ce_operate(modify_logits, mod_gts)
            tag_combine_loss += modify_loss
        return tag_combine_loss

    def forward(self, preds : dict, gts : dict, switch_mask : torch.Tensor = None, need_info : bool = False):
        # Binary Loss
        binary_preds, binary_gts = preds['binary'], gts['binary']
        binary_loss = self.cross_entropy(binary_preds, binary_gts)
        # Type Loss
        logits_iwo, logits_ip, logits_sc, logits_ill, logits_cm, logits_cr, logits_um = preds['type']
        gts_iwo, gts_ip, gts_sc, gts_ill, gts_cm, gts_cr, gts_um = gts['type'].T
        type_loss = self.cross_entropy(logits_iwo, gts_iwo) + self.cross_entropy(logits_ip, gts_ip)  + self.cross_entropy(logits_sc, gts_sc) + \
            self.cross_entropy(logits_cm, gts_cm) + self.cross_entropy(logits_ill, gts_ill) + self.cross_entropy(logits_cr, gts_cr) +self.cross_entropy(logits_um, gts_um)
        # Pointer Loss
        switch_preds, switch_gts = preds['switch'], gts['switch']
        # pointer_loss = self.pointer_loss(switch_preds, switch_gts, switch_mask)
        pointer_loss = self.pointer_loss(switch_preds, switch_gts, None) # TODO :Order Loss is forb
        # Tagger Loss
        tagger_preds, tagger_gts = preds['tagger'], gts['tagger']
        tagger_loss = self.gamma * self.tagger_loss(tagger_preds, tagger_gts)
        # Generate Loss
        mlm_logits, mlm_tgts = preds['generate'], gts['generate']
        output_mlm = self.softmax(mlm_logits)
        loss_mlm = self.nll_loss(output_mlm, mlm_tgts)
        # Gather Loss
        total_loss = binary_loss + type_loss + pointer_loss + tagger_loss +loss_mlm
        if need_info:
            loss_info = {
                'binary'  : binary_loss.item(),
                'type'    : type_loss.item(),
                'switch'  : pointer_loss.item(),
                'tagger'  : tagger_loss.item(),
                'generate': loss_mlm.item()
            }
            return total_loss, loss_info
        else:
            return total_loss