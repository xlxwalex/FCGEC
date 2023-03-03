from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import json
from utils import PointConverter, TextWash, data_filter, TYPE_MAP
import copy

class SwitchDataset(Dataset):
    def __init__(self, args, path : str, desc : str):
        super(SwitchDataset, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.desc = desc
        if hasattr(args, 'sp_map') and args.sp_map: self.spmap = True
        else: self.spmap = False
        # DATA PROCESSER
        self.error_number  = 0
        self.sentences, self.label   = self._read_csv(path)
        self.origin_label = copy.deepcopy(self.label)
        self.label = data_filter(self.sentences, self.label)
        self.point_seq, self.token, self.wd_idx, self.label = self._process_switch(self.sentences, self.label)

    def _read_csv(self, path):
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            labels.append(json.loads(ele[1]))
        return sentences, labels

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

    def _process_switch(self, sentences, labels):
        '''
        Process Switch Labels
        :param sentences: sentence list
        :param labels: label list
        :return: point list, token list, label
        '''
        point_seqs, wd_collect, post_labels, token_collection = [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]) if not self.spmap else TextWash.punc_wash_res(sentences[idx])[0])
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]) if not self.spmap else TextWash.punc_wash_res(sentences[idx]),
                'ops' : labels[idx],
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                pointer = PointConverter(self.args, auto=True, spmap=self.spmap, **kwargs)
                wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            except:
                import traceback
                print(traceback.print_exc())
                self.error_number += 1
                print(sentences[idx])
                continue
            if len(pointer.labels) > self.args.padding_size:
                self.error_number += 1
                continue
            point_seqs.append(pointer)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            post_labels.append(pointer.getlabel(offset=False))
        return  point_seqs, token_collection, wd_collect, post_labels

    def __getitem__(self, item):
        wid, label = self.wd_idx[item], self.label[item]
        ret = {
            'token' : wid,
            'label' : label
        }
        return ret

    def __len__(self):
        return len(self.label)

class SwitchDatasetWTTI(Dataset):
    def __init__(self, args, path : str, desc : str):
        super(SwitchDatasetWTTI, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.desc = desc
        # DATA PROCESSER
        self.sentences, self.types, self.label = self._read_csv(path)
        self.origin_label = copy.deepcopy(self.label)
        self.label = data_filter(self.sentences, self.label)
        self.point_seq, self.token, self.wd_idx, self.label = self._process_switch(self.sentences, self.label)

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

    def _process_switch(self, sentences, labels):
        '''
        Process Switch Labels
        :param sentences: sentence list
        :param labels: label list
        :return: point list, token list, label
        '''
        point_seqs, wd_collect, post_labels, token_collection = [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : labels[idx],
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                pointer = PointConverter(self.args, auto=True, **kwargs)
            except:
                print(sentences[idx])
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            pointer_label = pointer.getlabel(offset=False)
            if max(pointer_label) >= 150:
                continue
            point_seqs.append(pointer)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            post_labels.append(pointer.getlabel(offset=False))
        return  point_seqs, token_collection, wd_collect, post_labels

    def __getitem__(self, item):
        wid, type, label = self.wd_idx[item], self.types[item], self.label[item]
        ret = {
            'token' : wid,
            'type'  : type,
            'label' : label
        }
        return ret

    def __len__(self):
        return len(self.wd_idx)