import numpy as np
from argparse import Namespace
import operator as op
import torch
from utils.defines import TAGGER_MAP, INSERT_TAG, MODIFY_TAG, KEEP_TAG

class Metric(object):
    '''
    Base Modules of Metric Calculation
    '''
    def __init__(self, args : Namespace):
        self.args = args
        self.denom = 1e-8

    def __call__(self, gts, preds, mask : list) -> dict:
        raise NotImplementedError

    def _cal_token_level(self, gts, preds, mask: list):
        raise NotImplementedError

    def _cal_sentence_level(self, gts, preds, mask : list):
        raise NotImplementedError

class SwitchMetric_Spec():
    def __init__(self, args : Namespace, mode = 'all'):
        super(SwitchMetric_Spec, self).__init__()
        self.args = args
        self.mode = mode
        self.amax = args.padding_size
        self.use_lm = args.use_lm

    def __call__(self, wd_idxs : list, gts: list, preds: list, mask: list, out_put: bool = False) -> tuple:
        assert len(gts) == len(preds)
        pred_token = self._apply_switch_operator(wd_idxs, preds)
        truth_token = self._apply_switch_operator(wd_idxs, gts)
        token_acc = self._cal_token_level(truth_token, pred_token, mask)
        sentence_acc = self._cal_sentence_level(truth_token, pred_token, mask)
        if out_put is not True:
            return token_acc, sentence_acc
        else:
            return token_acc, sentence_acc, pred_token, truth_token

    def _apply_switch_operator(self, wd_idxs : list, switch_ops: list) -> list:
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
            # assert len(post_token) == np.sum(ori_token > 0)
            res.append(post_token)
        return res

    def _cal_token_level(self, gts : list, preds : list, mask : list) -> float:
        size = len(gts)
        assert len(preds) == size
        total_token = np.sum(np.array(mask))
        correct_token = 0
        for idx in range(size):
            for lidx in range(min(len(gts[idx]), len(preds[idx]))):
                if gts[idx][lidx] == preds[idx][lidx]: correct_token += 1
        return correct_token * 1. / total_token

    def _cal_sentence_level(self, gts : list, preds : list, mask : list) -> float:
        size = len(gts)
        assert len(preds) == size
        correct = 0
        for idx in range(size):
            if op.eq(gts[idx], preds[idx]):correct += 1
        return correct * 1. / size


class SwitchMetric(Metric):
    def __init__(self, args : Namespace, mode = 'all'):
        super(SwitchMetric, self).__init__(args)
        self.args = args
        self.mode = mode
        self.amax = args.padding_size
        self.use_lm = args.use_lm

    def __call__(self, gts : list, preds : list, mask : list) -> dict:
        '''
        Calculate Switch Metric
        :param gts: groud truth
        :param preds: preds label
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        assert len(gts) == len(preds)
        preds = self._repairlm_sep(preds, mask)
        if self.mode == 'all':
            ret['token'] = self._cal_token_level(gts, preds, mask)
            ret['sentence'] = self._cal_sentence_level(gts, preds)
        elif self.mode == 'token':
            ret['token'] = self._cal_token_level(gts, preds, mask)
        elif self.mode == 'sentence':
            ret['sentence'] = self._cal_sentence_level(gts, preds)
        else:
            raise Exception('SwitchMetric.__call__ occure some errors, invalid params `mode`.')
        return ret

    def _repairlm_sep(self, preds : list, mask : list):
        if self.use_lm:
            seq_lens = [np.where(mk == 0)[0][0] - 1 if mk[-2] != 1 else len(mk) - 2 if mk[-1] != 1 else len(mk) - 1 for mk in mask]
            for insid in range(len(seq_lens)): preds[insid][seq_lens[insid]] = 0
        return preds

    def _cal_token_level(self, gts : list, preds : list, mask : list) -> float:
        gts = np.clip(gts, a_min=0, a_max=self.amax)
        total_token = len(np.where(gts > 0)[0])
        externel_token = np.array(mask).size - total_token
        correct_token = np.sum(np.array(gts) == np.array(preds)) - externel_token
        return correct_token * 1. / total_token

    def _cal_sentence_level(self, gts : list, preds : list, mask : list = None) -> float:
        total_sentence = len(gts)
        gts = np.clip(gts, a_min=0, a_max=self.amax)
        correct_sentence = sum([1 if op.eq(gts[ins_idx].tolist(), preds[ins_idx].tolist()) else 0 for ins_idx in range(len(gts))])
        return correct_sentence * 1. / total_sentence

class TaggerMetric(Metric):
    def __init__(self, args : Namespace, mode = 'all', level = 'normal'):
        super(TaggerMetric, self).__init__(args)
        self.args = args
        self.mode = mode
        self.level = level # high/normal (whether ignore element not in MI & I)
        self.amax = args.padding_size
        self.use_lm = args.use_lm
        self.ins_tag = TAGGER_MAP[INSERT_TAG]
        self.mod_tag = TAGGER_MAP[MODIFY_TAG]
        assert level in ['normal', 'high']

    def __call__(self, gts : dict, preds : dict, mask : list) -> dict:
        '''
        Calculate Tagger Metric
        :param gts: groud truth (dict)
        :param preds: preds label (dict)
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        token_ret = self._cal_token_level(gts, preds, mask, self.level)
        ret['token'] = token_ret
        sentence_ret = self._cal_sentence_level(gts, preds, mask, self.level)
        ret['sentence'] = sentence_ret
        return ret

    def _cal_token_level(self, gts : dict, preds : dict, mask : list,  level: str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        ins_gts = np.clip(gts['insert'], a_min=0, a_max=np.max(gts['insert']))
        mod_gts = np.clip(gts['modify'], a_min=0, a_max=np.max(gts['modify']))
        tagger_preds = preds['tagger']
        ins_preds = preds['insert']
        mod_preds = preds['modify']
        # Calculate
        total_token = np.sum(np.array(mask))
        externel_token = np.array(mask).size - total_token
        batch_size = len(tagger_preds)
        # | - Tagger
        if level == 'normal':
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_gts[ins] > 0] == np.array(tagger_preds[ins])[tagger_gts[ins] > 0]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / total_token
        else:
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]] == np.array(tagger_preds[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / (len(np.where(np.array(tagger_preds) > TAGGER_MAP[KEEP_TAG])[0]) + self.denom)
        # | - Insert
        if level == 'normal':
            insert_index   = np.array(tagger_gts) == self.ins_tag
            correct_insert = np.sum(np.array(ins_gts)[insert_index] == np.array(ins_preds)[insert_index])
            total_insert   = len(np.where(tagger_gts == self.ins_tag)[0]) + self.denom
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            correct_insert = np.sum(np.array(ins_gts)[insert_index] == np.array(ins_preds)[insert_index])
            total_insert = len(np.where(tagger_gts == self.ins_tag)[0]) + self.denom
            # correct_insert = np.sum(np.array(ins_gts) == self.ins_tag) - externel_token
            # total_insert   = total_token
        insert_acc = correct_insert * 1. / total_insert
        # | - Modify
        if level == 'normal':
            modify_index   = np.array(tagger_gts) == self.mod_tag
            correct_modify = np.sum(np.array(mod_gts)[modify_index] == np.array(mod_preds)[modify_index])
            total_modify   = len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
        else:
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_modify = np.sum(np.array(mod_gts)[modify_index] == np.array(mod_preds)[modify_index])
            total_modify = len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
            # correct_modify = np.sum(np.array(mod_gts) == np.array(mod_preds)) - externel_token
            # total_modify   = total_token
        modify_acc = correct_modify * 1. / total_modify
        return {'tagger' : tagger_acc, 'insert' : insert_acc, 'modify' : modify_acc}

    def _cal_sentence_level(self, gts : dict, preds : dict, mask : list, level :str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        ins_gts = np.clip(gts['insert'], a_min=0, a_max=np.max(gts['insert']))
        mod_gts = np.clip(gts['modify'], a_min=0, a_max=np.max(gts['modify']))
        tagger_preds = preds['tagger']
        ins_preds = preds['insert']
        mod_preds = preds['modify']
        # Calculate
        total_sentence = len(gts['tagger'])
        # | - Tagger
        correct_tagger= sum([1 if op.eq(tagger_gts[ins_idx][tagger_gts[ins_idx]>0].tolist(), tagger_preds[ins_idx][tagger_gts[ins_idx]>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        tagger_acc = correct_tagger * 1. / total_sentence
        # | - Insert
        if level == 'normal':
            insert_index = np.array(tagger_gts) == self.ins_tag
            correct_insert = sum([1 if op.eq(ins_gts[ins_idx][insert_index[ins_idx]].tolist(), ins_preds[ins_idx][insert_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            correct_insert = sum([1 if op.eq(ins_gts[ins_idx][insert_index[ins_idx]].tolist(), ins_preds[ins_idx][insert_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
            #correct_insert = sum([1 if op.eq(ins_gts[ins_idx][ins_gts>0].tolist(), ins_preds[ins_idx][ins_gts>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        non_insert = sum([1 if np.max(ins_gts[ins_idx]) < 1 else 0 for ins_idx in range(total_sentence)])
        insert_acc = (correct_insert - non_insert) * 1. / (total_sentence - non_insert + self.denom) if non_insert != total_sentence else 1.0
        # | - Modify
        if level == 'normal':
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_modify = sum([1 if op.eq(mod_gts[ins_idx][modify_index[ins_idx]].tolist(), mod_preds[ins_idx][modify_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
        else:
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_modify = sum([1 if op.eq(mod_gts[ins_idx][modify_index[ins_idx]].tolist(), mod_preds[ins_idx][modify_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
            #correct_modify = sum([1 if op.eq(mod_gts[ins_idx][mod_gts > 0].tolist(), mod_preds[ins_idx][mod_gts > 0].tolist()) else 0 for ins_idx in range(total_sentence)])
        non_modify = sum([1 if np.max(mod_gts[ins_idx]) < 1 else 0 for ins_idx in range(total_sentence)])
        modify_acc = (correct_modify - non_modify) * 1. / (total_sentence - non_modify + self.denom) if non_modify != total_sentence else 1.0
        return {'tagger' : tagger_acc, 'insert' : insert_acc, 'modify' : modify_acc}

class TaggerMetricV2(Metric):
    def __init__(self, args : Namespace, mode = 'all', level = 'normal'):
        super(TaggerMetricV2, self).__init__(args)
        self.args = args
        self.mode = mode
        self.level = level # high/normal (whether ignore element not in MI & I)
        self.amax = args.padding_size
        self.use_lm = args.use_lm
        self.ins_tag = TAGGER_MAP[INSERT_TAG]
        self.mod_tag = TAGGER_MAP[MODIFY_TAG]
        assert level in ['normal', 'high']

    def __call__(self, gts : dict, preds : dict, mask : list) -> dict:
        '''
        Calculate Tagger Metric
        :param gts: groud truth (dict)
        :param preds: preds label (dict)
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        token_ret = self._cal_token_level(gts, preds, mask, self.level)
        ret['token'] = token_ret
        sentence_ret = self._cal_sentence_level(gts, preds, mask, self.level)
        ret['sentence'] = sentence_ret
        return ret

    def _cal_token_level(self, gts : dict, preds : dict, mask : list,  level: str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        insmod_gts = np.clip(gts['insmod'], a_min=0, a_max=np.max(gts['insmod']))
        tagger_preds = preds['tagger']
        insmod_preds = preds['insmod']
        # Calculate
        total_token = np.sum(np.array(mask))
        externel_token = np.array(mask).size - total_token
        batch_size = len(tagger_preds)
        # | - Tagger
        if level == 'normal':
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_gts[ins] > 0] == np.array(tagger_preds[ins])[tagger_gts[ins] > 0]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / total_token
        else:
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]] == np.array(tagger_preds[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / (len(np.where(np.array(tagger_preds) > TAGGER_MAP[KEEP_TAG])[0]) + self.denom)
        # | - InsMod
        if level == 'normal':
            insert_index   = np.array(tagger_gts) == self.ins_tag
            modify_index   = np.array(tagger_gts) == self.mod_tag
            correct_ins = np.sum(np.array(insmod_gts)[insert_index] == np.array(insmod_preds)[insert_index])
            correct_mod = np.sum(np.array(insmod_gts)[modify_index] == np.array(insmod_preds)[modify_index])
            correct_comb = correct_ins + correct_mod
            total_comb   = len(np.where(tagger_gts == self.ins_tag)[0])  + len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_ins  = np.sum(np.array(insmod_gts)[insert_index] == np.array(insmod_preds)[insert_index])
            correct_mod  = np.sum(np.array(insmod_gts)[modify_index] == np.array(insmod_preds)[modify_index])
            correct_comb = correct_ins + correct_mod
            total_comb   = len(np.where(tagger_gts == self.ins_tag)[0])  + len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
            # correct_insert = np.sum(np.array(ins_gts) == self.ins_tag) - externel_token
            # total_insert   = total_token
        comb_acc = correct_comb * 1. / total_comb
        return {'tagger' : tagger_acc, 'comb' : comb_acc}

    def _cal_sentence_level(self, gts : dict, preds : dict, mask : list, level :str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        insmod_gts = np.clip(gts['insmod'], a_min=0, a_max=np.max(gts['insmod']))
        tagger_preds = preds['tagger']
        insmod_preds = preds['insmod']
        # Calculate
        total_sentence = len(gts['tagger'])
        # | - Tagger
        correct_tagger= sum([1 if op.eq(tagger_gts[ins_idx][tagger_gts[ins_idx]>0].tolist(), tagger_preds[ins_idx][tagger_gts[ins_idx]>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        tagger_acc = correct_tagger * 1. / total_sentence
        # | - InsMod
        if level == 'normal':
            insert_index = np.array(tagger_gts) == self.ins_tag
            modify_index = np.array(tagger_gts) == self.mod_tag
            insmod_index = insert_index + modify_index
            correct_insmod = sum([1 if op.eq(insmod_gts[ins_idx][insmod_index[ins_idx]].tolist(), insmod_preds[ins_idx][insmod_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            modify_index = np.array(tagger_gts) == self.mod_tag
            insmod_index = insert_index + modify_index
            correct_insmod = sum([1 if op.eq(insmod_gts[idx][insmod_index[idx]].tolist(), insmod_preds[idx][insmod_index[idx]].tolist()) else 0 for idx in range(total_sentence)])
            #correct_insert = sum([1 if op.eq(ins_gts[ins_idx][ins_gts>0].tolist(), ins_preds[ins_idx][ins_gts>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        non_insmod = sum([1 if np.max(insmod_gts[idx]) < 1 else 0 for idx in range(total_sentence)])
        insmod_acc = (correct_insmod - non_insmod) * 1. / (total_sentence - non_insmod + self.denom) if non_insmod != total_sentence else 1.0
        return {'tagger' : tagger_acc, 'comb' : insmod_acc}

class GeneratorMetric(Metric):
    def __init__(self, args : Namespace, mode = 'all'):
        super(GeneratorMetric, self).__init__(args)
        self.args = args
        self.mode = mode
        self.amax = args.padding_size
        self.use_lm = args.use_lm

    def __call__(self, gts : list, preds : list, mask : list) -> dict:
        '''
        Calculate Switch Metric
        :param gts: groud truth
        :param preds: preds label
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        assert len(gts) == len(preds)
        if self.mode == 'all':
            ret['token'] = self._cal_token_level(gts, preds, mask)
            ret['sentence'] = self._cal_sentence_level(gts, preds, mask)
        elif self.mode == 'token':
            ret['token'] = self._cal_token_level(gts, preds, mask)
        elif self.mode == 'sentence':
            ret['sentence'] = self._cal_sentence_level(gts, preds, mask)
        else:
            raise Exception('GeneratorMetric.__call__ occure some errors, invalid params `mode`.')
        return ret

    def _cal_token_level(self, gts : list, preds : list, mask : list) -> float:
        #gts_token = np.clip(np.array(gts), a_min=0, a_max=1)
        total_token = np.array(gts).size
        correct_token = np.sum(np.array(gts) == np.array(preds))
        return correct_token * 1. / total_token

    def _cal_sentence_level(self, gts : list, preds : list, mask : list = None) -> float:
        mask = np.clip(np.array(mask), 0, 1)
        mask_length = np.sum(mask, axis=1).tolist()
        index = 0
        total_sentence = len(mask)
        correct_sentence = 0
        for elen in range(len(mask)):
            if op.eq(gts[index:index+mask_length[elen]], preds[index:index+mask_length[elen]]):
                correct_sentence += 1
            index += mask_length[elen]
        return correct_sentence * 1. / total_sentence