from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import json
from copy import copy
from utils import TYPE_MAP, PointConverter, TaggerConverter, TextWash, data_filter, tagger2generator, TAGGER_MAP, map_unk2word
from utils.data_utils import combine_insert_modify

class JointDataset(Dataset):
    def __init__(self, args, path : str, desc : str):
        super(JointDataset, self).__init__()
        self.args          = args
        # INITIALIZE
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.num_classes   = args.num_classes
        self.padding_size  = args.padding_size
        self.desc = desc
        # DATA PROCESSER
        self.sentences, self.label   = self._read_csv(path)
        if self.desc in ['train', 'valid', 'test']:
            self.operates = data_filter(self.sentences, self.label)
        # Switch Data
        self.error_ids = []
        self.point_seq, self.token, self.wd_idx, self.sw_label, self.unk_map = self._process_switch(self.sentences, self.operates)
        # Tagger Data & Generate Data
        self.tag_token, self.tagwd_idx = self._switch_tokens(self.point_seq, self.operates, self.token)
        self.tagger_seq, self.tokens, self.tagger_tokens, self.tagger_label, self.comb_labels, \
            self.gen_token, self.genwd_idx, self.tgt_mlm = self._process_tagger(self.sentences, self.operates)
        self.tagger_idx = self._tag2idx(self.tagger_label)
        print('>> Joint {} Dataset Has Been Processed'.format(desc))

    def _read_csv(self, path):
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            try:
                labels.append(json.loads(ele[1]))
            except:
                print(ele[0])
        return sentences, labels

    def _process_switch(self, sentences, labels):
        '''
        Process Switch Labels
        :param sentences: sentence list
        :param labels: label list
        :return: point list, token list, label
        '''
        point_seqs, wd_collect, post_labels, token_collection = [], [], [], []
        unk_map = []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset[Switch Part]'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : labels[idx],
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            unk_map.append(map_unk2word(tokens, sentences[idx]))
            try:
                pointer = PointConverter(self.args, auto=True, **kwargs)
            except:
                print(sentences[idx])
            lab = pointer.getlabel(offset=False)
            if max(lab) >= 150:
                print(sentences[idx])
                self.error_ids.append(idx)
                continue
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            point_seqs.append(pointer)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            post_labels.append(pointer.getlabel(offset=False))
        return  point_seqs, token_collection, wd_collect, post_labels, unk_map

    def _switch_tokens(self, pointer_seqs :list, operates : list, token_ls : list):
        tag_tokens, tagwd_idxs = [], []
        for idx in tqdm(range(len(pointer_seqs)), desc='Processing ' + self.desc + ' Dataset[Switch Trans_Tokens]'):
            pointer = pointer_seqs[idx]
            operate = operates[idx]
            tokens  = token_ls[idx]
            if 'Switch' not in operate:
                tag_tokens.append(tokens)
            else:
                tag_tokens.append([ele.token for ele in pointer.point_sequence])
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            tagwd_idxs.append(wd_idxs)
        return tag_tokens, tagwd_idxs

    def _tag2idx(self, tagger_labels : list):
        '''
        Convert Tagger Labels 2 Index based on Defines
        :param tagger_labels: Tagger labels (list)
        :return: Tagger Labels(index map) (list)
        '''
        tagidxs = [[TAGGER_MAP[ele] for ele in ins] for ins in tagger_labels]
        return tagidxs

    def _preprocess_modify(self, ops: dict):
        '''
        Pre-tokenize modify labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = self.tokenizer.tokenize(labstr)
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = self.tokenizer.tokenize(labstr)
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    def _preprocess_gendata(self, ops: dict):
        '''
        Pre-tokenize modify labels and insert labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(labstr))
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(labstr))
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    def _process_tagger(self, sentences : list, labels : list) -> tuple:
        '''
        Process Tagger Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tagger_label, comb_labels = [], [], [], [], []
        gen_tokens, genwd_idxs, tgt_mlms = [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset[Tagger/Gen Part]'):
            if idx in self.error_ids:
                continue
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : self._preprocess_gendata(labels[idx]),
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                tagger = TaggerConverter(self.args, auto=True, **kwargs)
            except:
                print(sentences[idx])
            label_comb = tagger.getlabel(types='dict')
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            tagger_seqs.append(tagger)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
            tagger_label.append(label_comb['tagger'])
            gen_token, gen_label = tagger2generator(tokens, label_comb['tagger'], label_comb['mask_label'])
            comb_labels.append(comb_label)
            #genwd_idxs.append(wd_idxs)
            genwd_idxs.append(self.tokenizer.convert_tokens_to_ids(gen_token))
            gen_tokens.append(gen_token)
            tgt_mlms.append(gen_label)
        return tagger_seqs, token_collection, wd_collect, tagger_label, comb_labels, gen_tokens, genwd_idxs, tgt_mlms

    def __getitem__(self, item):
        wid, wid_tag, wid_gen = self.wd_idx[item], self.tagwd_idx[item], self.genwd_idx[item]
        mlm_label = self.tgt_mlm[item]
        tag_label, comb_label = self.tagger_idx[item], self.comb_labels[item]
        sw_label = self.sw_label[item]
        ret = {
            'wid_ori'   : wid,
            'wid_tag'   : wid_tag,
            'wid_gen'   : wid_gen,

            'tag_label' : tag_label,
            'comb_label': comb_label,

            'swlabel'   : sw_label,
            'mlmlabel'  : mlm_label
        }
        return ret

    def __len__(self):
        return len(self.sw_label)