from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import json
from utils import TaggerConverter, TextWash, switch_convertor, tagger2generator
from copy import copy

class GeneratorDataset(Dataset):
    def __init__(self, args, path : str, desc : str, token_ext : list = None, label_ext : list = None):
        super(GeneratorDataset, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.desc = desc
        self.error_number = 0
        # DATA PROCESSER
        sentences, labels  = self._read_csv(path)
        sentences, labels = self._unpack(sentences, labels)
        self.sentences, self.labels = self._extract_gendata(sentences, labels)
        if desc != 'test' and token_ext is None and label_ext is None:
            self.tagger_seq, self.token, self.wd_idx, self.mlm_labels = self._process_generator(self.sentences, self.labels)
        else:
            self.wd_idx = token_ext
            self.mlm_labels = label_ext
            self.labels = label_ext

    def _read_csv(self, path):
        type_flag = False
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        if data.shape[1] == 3: type_flag = True
        for ele in data:
            sentences.append(ele[0])
            labels.append(json.loads(ele[1])) if not type_flag else labels.append(json.loads(ele[2]))
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

    def _extract_gendata(self, sentences : list, labels : list) -> tuple:
        '''
        Extract Data that Contains I / MI / M
        :param sentences: sentences collection list
        :param labels: labels list
        :return: extracted sentence, label
        '''
        post_sentences, post_labels = [], []
        for index in range(len(sentences)):
            sentence, label = sentences[index], labels[index]
            if 'Switch' in label:
                try:
                    sentence = switch_convertor(sentence, label['Switch'])
                except:
                    print(sentence)
            if 'Insert' in label or 'Modify' in label:
                if 'Modify' in label:
                    mod_op = label['Modify']
                    MI_flags = [True if 'INS' in mod['tag'] or '+' not in mod['tag'] else False for mod in mod_op]
                    if True not in MI_flags:
                        continue
                post_sentences.append(sentence)
                post_labels.append(label)
            else:
                continue
        return post_sentences, post_labels

    def _preprocess_insert(self, ops : dict):
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

    def _process_generator(self, sentences : list, labels : list) -> tuple:
        '''
        Process Generator Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tgt_mlms = [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            try:
                kwargs = {
                    'sentence' : TextWash.punc_wash(sentences[idx]),
                    'ops' : self._preprocess_insert(labels[idx]),
                    'token' : token
                }
            except:
                print(sentences[idx])
            tokens = [self.CLS] + token + [self.SEP]
            try:
                tagger = TaggerConverter(self.args, auto=True, **kwargs)
                label_comb = tagger.getlabel(types='dict')
                tags, mask_label = label_comb['tagger'], label_comb['mask_label']
                tokens, label = tagger2generator(tokens, tags, mask_label)
                wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            except:
                print(sentences[idx])
                self.error_number += 1
                continue
            tagger_seqs.append(tagger)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            tgt_mlms.append(label)
        return tagger_seqs, token_collection, wd_collect, tgt_mlms

    def __getitem__(self, item):
        wid, mlm_label = self.wd_idx[item], self.mlm_labels[item]
        ret = {
            'token'  : wid,
            'label'  : mlm_label
        }
        return ret

    def __len__(self):
        return len(self.wd_idx)
