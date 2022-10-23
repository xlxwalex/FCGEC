from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import json
from utils import TaggerConverter, TextWash, TAGGER_MAP, data_filter, TYPE_MAP
from utils.data_utils import combine_insert_modify
from copy import copy

class TaggerDataset(Dataset):
    def __init__(self, args, path : str, desc : str, token_ext : list = None):
        super(TaggerDataset, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.error_number  = 0
        self.desc = desc
        self.sentences, self.label = self._read_csv(path)
        if self.desc != 'test' or token_ext is None:
            # DATA PROCESSER
            # self.sentences, self.label = self._unpack(sentences, label)
            self.label = data_filter(self.sentences, self.label)
            self.tagger_seq, self.token, self.wd_idx, self.tagger_label, self.comb_label = self._process_tagger(self.sentences, self.label)
            self.tagger_idx = self._tag2idx(self.tagger_label)
        else:
            self.wd_idx = token_ext

    def _read_csv(self, path):
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            labels.append(json.loads(ele[1]))
        return sentences, labels

    def _tag2idx(self, tagger_labels : list):
        '''
        Convert Tagger Labels 2 Index based on Defines
        :param tagger_labels: Tagger labels (list)
        :return: Tagger Labels(index map) (list)
        '''
        tagidxs = [[TAGGER_MAP[ele] for ele in ins] for ins in tagger_labels]
        return tagidxs

    def _unpack(self, sentences, labels):
        '''
        Unpack multi-operator samples to sigle-operator samples (Expand)
        :param sentence: sentence list
        :param label: label list
        :return: expanded sentence, label
        '''
        assert len(sentences) == len(labels)
        unpack_sentences, unpack_labels = [], []
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            label = labels[idx]
            for ele in label:
                unpack_sentences.append(sentence)
                unpack_labels.append(ele)
        return unpack_sentences, unpack_labels

    def _preprocess_modify(self, ops : dict):
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

    def _process_tagger(self, sentences : list, labels : list, token_ext : list=None) -> tuple:
        '''
        Process Tagger Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tagger_label, comb_labels = [], [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : self._preprocess_modify(labels[idx]),
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                tagger = TaggerConverter(self.args, auto=True, **kwargs)
                label_comb = tagger.getlabel(types='dict')
                comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
                # if max(comb_label) > 10:
                #     self.error_number += 1
                #     continue
            except:
                self.error_number += 1
                print(sentences[idx])
                continue
            # label_comb = tagger.getlabel(types='dict')
            # comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            tagger_seqs.append(tagger)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            tagger_label.append(label_comb['tagger'])
            comb_labels.append(comb_label)
        return tagger_seqs, token_collection, wd_collect, tagger_label, comb_labels

    def __getitem__(self, item):
        if self.desc != 'test':
            wid, tagger, comb = self.wd_idx[item], self.tagger_idx[item], self.comb_label[item]
            ret = {
                'token'  : wid,
                'tagger' : tagger,
                'comb'   : comb
            }
        else:
            wid = self.wd_idx[item]
            ret = {
                'token' : wid
            }
        return ret

    def __len__(self):
        return len(self.wd_idx)

class TaggerDatasetTTI(Dataset):
    def __init__(self, args, path : str, desc : str):
        super(TaggerDatasetTTI, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.desc = desc
        self.sentences, self.types, self.label = self._read_csv(path)
        # DATA PROCESSER
        # self.sentences, self.label = self._unpack(sentences, label)
        self.label = data_filter(self.sentences, self.label)
        self.tagger_seq, self.token, self.wd_idx, self.tagger_label, self.comb_label = self._process_tagger(self.sentences, self.label)
        self.tagger_idx = self._tag2idx(self.tagger_label)

    def _read_csv(self, path):
        sentences, types, labels = [], [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            types.append(self._get_type_label(ele[1]))
            labels.append(json.loads(ele[2]))
        return sentences, types, labels

    def _get_type_label(self, labels : str):
        label = [0] * len(TYPE_MAP.keys())
        labels = labels.split(';')
        if 'UNK' in labels: return None
        for elab in labels:
            if elab == '*':return None
            label[TYPE_MAP[elab]] = 1
        return label

    def _tag2idx(self, tagger_labels : list):
        '''
        Convert Tagger Labels 2 Index based on Defines
        :param tagger_labels: Tagger labels (list)
        :return: Tagger Labels(index map) (list)
        '''
        tagidxs = [[TAGGER_MAP[ele] for ele in ins] for ins in tagger_labels]
        return tagidxs

    def _unpack(self, sentences, labels):
        '''
        Unpack multi-operator samples to sigle-operator samples (Expand)
        :param sentence: sentence list
        :param label: label list
        :return: expanded sentence, label
        '''
        assert len(sentences) == len(labels)
        unpack_sentences, unpack_labels = [], []
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            label = labels[idx]
            for ele in label:
                unpack_sentences.append(sentence)
                unpack_labels.append(ele)
        return unpack_sentences, unpack_labels

    def _preprocess_modify(self, ops : dict):
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

    def _process_tagger(self, sentences : list, labels : list, token_ext : list=None) -> tuple:
        '''
        Process Tagger Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tagger_label, comb_labels = [], [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : self._preprocess_modify(labels[idx]),
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
            comb_labels.append(comb_label)
        return tagger_seqs, token_collection, wd_collect, tagger_label, comb_labels

    def __getitem__(self, item):
        wid, tagger, comb, type = self.wd_idx[item], self.tagger_idx[item], self.comb_label[item], self.types[item]
        ret = {
            'token'  : wid,
            'tagger' : tagger,
            'comb'   : comb,
            "type"   : type
        }
        return ret

    def __len__(self):
        return len(self.label)