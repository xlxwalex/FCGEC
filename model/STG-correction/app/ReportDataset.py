from torch.utils.data import Dataset
from transformers import BertTokenizer
from argparse import Namespace
from utils import TAGGER_MAP, MODIFY_TAG, INSERT_TAG, MOIFY_ONLY_TAG, KEEP_TAG, DELETE_TAG, MODIFY_DELETE_TAG, MASK_SYMBOL

class ReportDataset(Dataset):
    def __init__(self, args : Namespace, sentences : list):
        super(ReportDataset, self).__init__()
        # INITIALIZE
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.args          = args
        self.tokenizer     = BertTokenizer.from_pretrained(self.args.lm_path, cache_dir='./.cache')
        self.num_classes   = self.args.num_classes
        self.padding_size  = self.args.padding_size
        self._phase        = "binary"
        # DATA PROCESSER
        self.sentences     = sentences
        self.wd_idx        = self._word2idx(self.sentences)
        self.binary_label  = None
        self.wd_tag_idx    = None
        self.tagger_label  = None
        self.wd_gen_idx    = None
        self.split_idx     = []
        self.generate_mask = []

    def _word2idx(self, sentences : list) -> list:
        wids = [self.tokenizer.convert_tokens_to_ids(
            [self.CLS] + self.tokenizer.tokenize(sentences[idx]) + [self.SEP]
        ) for idx in range(len(sentences))]
        return wids

    def binary(self):
        self._phase = "binary"

    def tagger(self, switch_pointer: list = None, switch_flag: list = None):
        self.switch_idx = switch_pointer
        self.switch_flag  = switch_flag
        self._phase = "tagger"
        if self.switch_flag != None:
            assert len(self.switch_flag) == len(self.wd_idx)
            self.wd_tag_idx = [self.switch_idx[idx] if self.switch_flag[idx] else self.wd_idx[idx] for idx in range(len(self.switch_flag))]
        else:
            self.wd_tag_idx = self.wd_idx

    def generate(self, tagger : list = None):
        self.tagger_label = tagger
        self._phase = "generate"
        if self.tagger_label is None:
            raise Exception("[ReportDataset] Generator Phase need tagger information.")
        tag_seq, tgt_mask = tagger
        filter_group = [(tag_seq[idx], tgt_mask[idx]) for idx in range(len(self.wd_tag_idx)) if 1 in tgt_mask[idx]]
        self.wd_gen_idx = [filter_group[idx][0] for idx in range(len(filter_group))]
        self.gen_mask = [filter_group[idx][1] for idx in range(len(filter_group))]
        self.filter_flag = [False if 1 in tgt_mask[idx] else True for idx in range(len(self.wd_tag_idx))]

    def _pad_mask(self, tagger : list, mitag : list, itag : list):
        if len(self.wd_tag_idx) == 0: return []
        ori_text = [self.tokenizer.convert_ids_to_tokens(self.wd_gen_idx[idx]) for idx in range(len(self.wd_gen_idx))]
        post_wd_idx = []
        mask_seq = []
        split_seq = []
        for idx in range(len(self.wd_gen_idx)):
            wd_idx = []
            mask, split = [], []
            mod, ins = 0, 0
            cur_text = ori_text[idx]
            tag = tagger[idx]
            for iidx in range(len(cur_text)):
                if tag[iidx] == TAGGER_MAP[KEEP_TAG]:
                    wd_idx.append(cur_text[iidx])
                    mask.append(0)
                elif tag[iidx] in [TAGGER_MAP[DELETE_TAG], TAGGER_MAP[MODIFY_DELETE_TAG]]:
                    continue
                elif tag[iidx] == TAGGER_MAP[MOIFY_ONLY_TAG]:
                    if iidx > 0 and tag[iidx-1] != TAGGER_MAP[MOIFY_ONLY_TAG]:
                        split.append(iidx)
                    mask.append(1)
                    wd_idx.append(MASK_SYMBOL)
                elif tag[iidx] == TAGGER_MAP[INSERT_TAG]:
                    wd_idx.append(cur_text[iidx])
                    mask.append(0)
                    split.append(iidx + 1)
                    mask.extend([1] * itag[idx][ins])
                    wd_idx.extend([MASK_SYMBOL] * itag[idx][ins])
                    ins += 1
                elif tag[iidx] == TAGGER_MAP[MODIFY_TAG]:
                    split.append(iidx)
                    mask.extend([1] * (mitag[idx][mod]+1))
                    wd_idx.extend([MASK_SYMBOL] * (mitag[idx][mod]+1))
                    mod += 1
                else:
                    continue
            mask_seq.append(mask)
            split_seq.append(split)
            post_wd_idx.append(wd_idx)
        return [self.tokenizer.convert_tokens_to_ids(ele) for ele in post_wd_idx], mask_seq, split_seq

    def __getitem__(self, item):
        if self._phase == 'binary':
            wid = self.wd_idx[item]
            ret = {
                'token' : wid,
            }
            return ret
        elif self._phase == 'tagger':
            wid = self.wd_tag_idx[item]
            ret = {
                'token': wid,
            }
            return ret
        elif self._phase == 'generate':
            wid, mask = self.wd_gen_idx[item], self.gen_mask[item]
            ret = {
                'token' : wid,
                'mask'  : mask
            }
            return ret
        else:
            raise Exception("[ReportDataset] Occured an error, param `phase` is not correct.")

    def __len__(self):
        if self._phase == 'binary':
            return len(self.sentences)
        elif self._phase == 'tagger':
            return len(self.wd_tag_idx)
        elif self._phase == 'generate':
            return len(self.wd_gen_idx)
        else:
            raise Exception("[ReportDataset] Occured an error, param `phase` is not correct.")