# Import Libs
import numpy as np
import torch
from utils.Point import Point
from copy import copy

def switch_convertor(sentence : str, switch_op : list):
    assert len(sentence) == len(switch_op)
    convert_tokens = [sentence[index] for index in switch_op]
    return ''.join(convert_tokens)

class Converter(object):
    """Base Converter Class"""

    def __init__(self, args):
        '''
        Base Converter Class
        :param args: Work Prams
        '''
        self.origins = []
        self.labels = []

    def convert_point(self, sentence : str, ops : dict, **kwargs):
        '''
        Convert Operators 2 Labels
        :param ops: operator dict (json format)
        :return:
        '''
        raise NotImplementedError

    def convert_tagger(self, sentence :str, ops : dict, **kwargs):
        raise NotImplementedError

    def get_ordersum(self):
        raise NotImplementedError

    def __len__(self):
        '''
        Get Length of Converter
        :return: Pointer Length
        '''
        return len(self.labels)

    def __repr__(self):
        '''
        Print Operator descriptions
        '''
        raise NotImplementedError

    def getlabel(self, types = "list"):
        raise NotImplementedError

    def _getlabel(self, types = "list"):
        '''
        Return Labels of Converter
        :param types: label type ["list", "numpy", "tensor"]
        :return: labels with specific format
        '''
        if len(self.labels) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        if types == "list":
            return self.labels
        elif types == "numpy":
            try:
                return np.array(self.labels)
            except:
                raise ValueError('Label should be an legal, not {}'.format(type(self.labels)))
        elif types == "tensor":
            try:
                return torch.from_numpy(np.array(self.labels))
            except:
                raise ValueError('Label should be an legal, not {}'.format(type(self.labels)))
        else:
            raise Exception('params types %s not exist, please check.' % types)

class PointConverter(Converter):
    """Converter from training target Switch Ops into pointor format."""

    def __init__(self, args, auto : bool = False, **kwargs):
        super(PointConverter, self).__init__(args)
        self.point_sequence = []
        self.post_sentence = ""
        self.origin_sentence = ""
        self.use_lm = False
        self.p2next = True
        try:
            if args.use_lm: self.use_lm = True
        except: self.use_lm = False
        try:
            if args.p2next is not True: self.p2next = False
        except: self.use_lm = False
        if auto:
            if 'sentence' in kwargs.keys() and 'ops' in kwargs.keys():
                kwargs_copy = copy(kwargs)
                del kwargs_copy['sentence']
                del kwargs_copy['ops']
                self.convert_point(sentence=kwargs['sentence'], ops=kwargs['ops'], **kwargs_copy)
            else:
                raise Exception("param `auto` is true, but param `sentence` or `ops` not found")

    def _convert_point_lm(self, sentence : str, ops : dict, token : list) -> tuple:
        '''
        Convert PLM Mechanism 2 PointConvertor Format
        :param sentence: original sentence (str)
        :param ops: operator description (dict)
        :param token: tokens (list)
        :return: post_sentence (list), post_ops (dict)
        '''
        self.origin_sentence = sentence if isinstance(sentence, str) else ''.join(sentence)
        post_sentence, post_ops = [], {'Switch' : []}
        pos_map = {-1 :-1}
        tpidx = 0
        for eidx in range(len(token)):
            if token[eidx] == '[CLS]' or token[eidx] == '[SEP]':
                continue
            if sentence[tpidx] == token[eidx] or token[eidx] == '[UNK]':
                post_sentence.append(token[eidx])
                pos_map[tpidx] = eidx
                tpidx += 1
            else:
                tmptoken = token[eidx]
                if tmptoken.startswith("#"):
                    tmptoken = token[eidx].replace('#', '')
                if tmptoken == sentence[tpidx] or tmptoken == '[UNK]':
                    post_sentence.append(token[eidx])
                    pos_map[tpidx] = eidx
                    tpidx += 1
                else:
                    tlen = len(tmptoken)
                    for cidx in range(tlen):
                        if tmptoken[cidx] == sentence[tpidx + cidx]:
                            pos_map[tpidx + cidx] = eidx
                        else:
                            raise Exception("The sample can not be convert to token case.")
                    tpidx += tlen
                    post_sentence.append(tmptoken)
        switch_ops = [-1] + ops['Switch']
        try:
            post_ops['Switch'] = [pos_map[switch_ops[eidx]] for eidx in range(1, len(switch_ops)) if pos_map[switch_ops[eidx]] != pos_map[switch_ops[eidx - 1]]]
        except: print('[{}] Sentence Error.'.format(sentence))
        assert len(token) == len(post_ops['Switch'])
        return post_sentence, post_ops

    def convert_point(self, sentence : str, ops : dict, **kwargs):
        self.origins = list(range(len(sentence)))
        if 'Switch' not in ops.keys():
            if self.use_lm:
                self.point_sequence.append(Point(0, '[CLS]'))
                if 'token' in kwargs.keys():
                    sentence = kwargs['token']
                self.origins = list(range(len(sentence)))
                self.point_sequence += [Point(ele+1, sentence[ele]) for ele in self.origins]
                self.point_sequence.append(Point(len(self.point_sequence), '[SEP]'))
            else:
                self.point_sequence = [Point(ele, sentence[ele]) for ele in self.origins]
        else:
            if 'token' in kwargs.keys():
                sentence, ops = self._convert_point_lm(sentence, ops, kwargs['token'])
            self.point_sequence = self._convert_point(sentence, ops['Switch'])
        self.post_sentence = ''.join([ele.token for ele in self.point_sequence])
        self.labels = [(ele.point_index, ele.offset) for ele in self.point_sequence]

    def _convert_p2next_label(self, ori_labels : list):
        '''
        Convert labels to p2next version
        :param ori_labels: origin label format
        :return: next version label
        '''
        fl_map = {}
        labels = [-1] * len(ori_labels)
        for eidx in range(len(ori_labels) - 1):
            fl_map[ori_labels[eidx]] = ori_labels[eidx + 1]
        fl_map[len(ori_labels) - 1] = -1
        for eidx in range(len(ori_labels)):
            labels[eidx] = fl_map[eidx]
        return labels

    def _convert_point(self, sentence : str, op : list) -> list:
        """
        Convert Switch ops 2 Labels
        :param sentence: sentence of sample
        :param op: Switch Op List
        :return: point_sequence
        """
        index_map_inv = dict(zip(op, list(range(len(sentence)))))
        sequence = []
        if self.use_lm:
            sequence.append(Point(0, '[CLS]', 0))
            for ele in op:
                sequence.append(Point(ele + 1, sentence[ele], abs(ele - index_map_inv[ele])))
            sequence.append(Point(len(sequence), '[SEP]', 0))
        else:
            for ele in op:
                sequence.append(Point(ele, sentence[ele], abs(ele - index_map_inv[ele])))
        return sequence

    def getlabel(self, types = "list", offset : bool = True):
        if len(self.point_sequence) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        if offset:
            if types != "list":
                raise Exception("PointConvertor not support %s type when you need offset." % types)
            return self.labels
        else:
            # Pointer2Next Version
            label = [ele[0] for ele in self.labels]
            if self.p2next and self.use_lm:
                label = self._convert_p2next_label(label)
            if types == "list":
                return label
            elif types == "numpy":
                return np.array(label)
            elif types == "tensor":
                return torch.from_numpy(np.array(label))
            else:
                raise Exception("PointConvertor only support ['list', 'numpy', 'tensor'] type")


    def convert_tagger(self, sentence :str, ops : dict, **kwargs):
        raise Exception("PointConvertor not support %s method" % "convert_tagger")

    def get_ordersum(self, need_seq : bool = False):
        '''
        Get Order for Regularization
        :param need_seq:  return order_seq or not
        :return: sum of order_seq
        '''
        order_2nd = []
        if len(self.labels) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        for idx in range(1, len(self.labels)):
            order_2nd.append(abs(self.labels[idx][-1] - self.labels[idx-1][-1]))
        if need_seq:
            return sum(order_2nd), order_2nd
        else:
            return sum(order_2nd)

    def __repr__(self):
        obreturn = ''
        obreturn += ('>>> PointConvertor Elements : ' + '\n')
        if len(self.labels) == 0 or len(self.point_sequence) == 0:
            obreturn += 'Null object'
            return obreturn
        obreturn += 'Original sentence : {}\n'.format(self.origin_sentence)
        obreturn += ('> position index :' + '\n')
        obreturn += (', '.join([str(ele) for ele in self.getlabel(offset=False)]) + '\n')
        obreturn += ('> convert sentence :' + '\n')
        obreturn += (''.join([ele.token for ele in self.point_sequence]) + '\n')
        obreturn += ('> index offset :' + '\n')
        obreturn += (', '.join([str(ele[1]) for ele in self.labels]) + '\n')
        obreturn += ('> Sum offset : {}\n'.format(sum([ele[1] for ele in self.labels])))
        order_sum, order_seq = self.get_ordersum(need_seq=True)
        obreturn += ('>> Order Sequence :\n' + ', '.join([str(ele) for ele in order_seq]) + '\n')
        obreturn += ('> Sum 2nd-order (Regular): {}'.format(order_sum))
        return obreturn

class TaggerConverter(Converter):
    """Converter from training target into Tagger format."""

    def __init__(self, args, auto : bool = False, **kwargs):
        super(TaggerConverter, self).__init__(args)
        self.tagger_sequence = []
        self.mask_label = {}
        self.ins_label = []
        self.mod_label = []
        self.post_tokens = []
        self.origin_sentence = ""
        self.use_lm = False
        self.ignore_index = args.ignore_val
        try:
            if args.use_lm: self.use_lm = True
        except:
            self.use_lm = False
        if auto:
            if 'sentence' in kwargs.keys() and 'ops' in kwargs.keys():
                kwargs_copy = copy(kwargs)
                del kwargs_copy['sentence']
                del kwargs_copy['ops']
                self.convert_tagger(sentence=kwargs['sentence'], ops=kwargs['ops'], **kwargs_copy)
            else:
                raise Exception("param `auto` is true, but param `sentence` or `ops` not found")

    def convert_point(self, sentence: str, ops: dict, **kwargs):
        raise Exception("PointConvertor not support %s method" % "convert_point")

    def getlabel(self, types="list") -> dict:
        if len(self.labels) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        if types != "dict":
            raise Exception("TaggerConverter not support %s type" % types)
        return self.labels

    def _convert_tagger_lm(self, sentence : str, ops : dict, token : list) -> tuple:
        '''
        Convert PLM Mechanism 2 Tagger ConvertorFormat
        :param sentence: original sentence(str)
        :param ops: operator description(dict)
        :param token: tokens(list)
        :return: post_sentence(list), post_ops(dict)
        '''
        self.origin_sentence = sentence if isinstance(sentence, str) else ''.join(sentence)
        post_sentence, post_ops = [], {}
        pos_map, pos_map_inv = {-1 :-1}, {-1 : -1}
        tpidx = 0
        for eidx in range(len(token)):
            if token[eidx] == '[CLS]' or token[eidx] == '[SEP]':
                continue
            if sentence[tpidx] == ' ':
                tpidx += 1
            if sentence[tpidx] == token[eidx] or token[eidx] == '[UNK]':
                post_sentence.append(token[eidx])
                pos_map[tpidx] = eidx
                pos_map_inv[eidx] = [tpidx]
                tpidx += 1
            else:
                tmptoken = token[eidx]
                if tmptoken.startswith("#"):
                    tmptoken = token[eidx].replace('#', '')
                if tmptoken == sentence[tpidx] or tmptoken == '[UNK]':
                    post_sentence.append(token[eidx])
                    pos_map[tpidx] = eidx
                    pos_map_inv[eidx] = [tpidx]
                    tpidx += 1
                else:
                    tlen = len(tmptoken)
                    pos_map_inv[eidx] = [tpidx]
                    tp_cidx = 0
                    for cidx in range(tlen):
                        if tmptoken[cidx] == sentence[tpidx + cidx]:
                            pos_map[tpidx + cidx] = eidx
                            tp_cidx = cidx
                        else:
                            raise Exception("The sample can not be convert to token case.")
                    pos_map_inv[eidx].append(tpidx + tp_cidx)
                    tpidx += tlen
                    post_sentence.append(tmptoken)
        if 'Delete' in ops:
            delete_op = []
            for didx in ops['Delete']:
                delete_op.append(pos_map[didx])
            post_ops['Delete'] = delete_op
        if 'Insert' in ops:
            insert_op = []
            for ins in ops['Insert']:
                nins = {}
                pos = ins['pos']
                nins['pos'] = pos_map[pos]
                nins['tag'] = ins['tag']
                nins['label'] = ins['label']
                nins['label_token'] = ins['label_token']
                insert_op.append(nins)
            post_ops['Insert'] = insert_op
        if 'Modify' in ops:
            modify_op = []
            for mod in ops['Modify']:
                nmod_op = {}
                pos, tag = mod['pos'], mod['tag']
                nmod_op['pos'] = pos_map[pos]
                tag_o = eval(tag.split('+')[0].split('_')[-1])
                label_length = len(mod['label_token'])
                e_pos = pos_map[pos + tag_o - 1]
                s_pos = pos_map[pos]
                cur_len = e_pos - s_pos + 1
                if cur_len == label_length:
                    nmod_op['tag'] = 'MOD_' + str(cur_len)
                else:
                    delta_len = label_length - cur_len
                    if delta_len > 0: nmod_op['tag'] = 'MOD_' + str(cur_len) + '+INS_' + str(delta_len)
                    else: nmod_op['tag'] = 'MOD_' + str(cur_len) + '+DEL_' + str(-delta_len)
                nmod_op['label'] = mod['label']
                nmod_op['label_token'] = mod['label_token']
                modify_op.append(nmod_op)
            post_ops['Modify'] = modify_op
        return post_sentence, post_ops

    def apply_tagger(self, tokens:list, tagger : list, mask_label : list):
        post_tokens = []
        for tid in range(len(tokens)):
            if tagger[tid] == 'K':
                post_tokens.append(tokens[tid])
            elif tagger[tid] in ['D', 'MD']:
                continue
            elif tagger[tid] == 'I':
                post_tokens.append(tokens[tid])
                post_tokens.extend(mask_label[tid])
            elif tagger[tid] == 'MI':
                post_tokens.append(mask_label[tid][0])
                post_tokens.extend(mask_label[tid][1:])
            elif tagger[tid] == 'M':
                post_tokens.append(mask_label[tid])
        return post_tokens

    def convert_tagger(self, sentence :str, ops : dict, **kwargs):
        self.origins = list(range(len(sentence)))
        self.origin_sentence = sentence
        # Preprocess LM
        if 'token' in kwargs.keys():
            sentence, ops = self._convert_tagger_lm(sentence, ops, kwargs['token'])
        self.tagger_sequence, self.mask_label, self.ins_label, self.mod_label = self._convert_tagger(sentence, ops)
        self.labels = {}
        self.labels['tagger'] = self.tagger_sequence
        self.labels['mask_label'] = self.mask_label
        self.labels['ins_label'] = self.ins_label
        self.labels['mod_label'] = self.mod_label
        self.post_tokens = self.apply_tagger(['[CLS]'] + kwargs['token'] + ['[SEP]'], self.tagger_sequence, self.mask_label)

    def _convert_tagger(self, post_sentence: str, ops: dict) -> tuple:
        '''
        Convert Tagger ops 2 Labels
        :param post_sentence: The sentence after Tagger operator
        :param ops: tagger_label, ins_label, mod_label, mask_label (list)
        '''
        if self.use_lm and isinstance(post_sentence, list):
            post_sentence = ['[CLS]'] + post_sentence + ['[SEP]']
        tagger = ['K'] * len(post_sentence)
        ins_label = [self.ignore_index] * len(post_sentence)
        mod_label = [self.ignore_index] * len(post_sentence)
        mask_label = {}
        # Delete Operator
        if 'Delete' in ops.keys():
            for ele in ops['Delete']:
                tagger[ele + 1] = 'D'
        # Insert Operator
        if 'Insert' in ops.keys():
            for inop in ops['Insert']:
                tagger[inop['pos'] + 1] = 'I'
                ins_label[inop['pos'] + 1] = eval(inop['tag'].split('_')[-1])
                mask_label[inop['pos'] + 1] = inop['label_token']
        # Modify Operator
        if 'Modify' in ops.keys():
            for moop in ops["Modify"]:
                oplen = moop['tag']
                if '+' not in oplen:
                    oplen = eval(oplen.split('_')[-1])
                    for idx in range(oplen):
                        tagger[moop['pos'] + idx + 1] = 'M'
                        mask_label[moop['pos'] + idx + 1] = moop['label_token'][idx]
                else:
                    # Contain + in op
                    if 'DEL' in moop['tag']:
                        oplen = len(moop['label_token'])
                        opdel = eval(moop['tag'].split('+')[-1].split('_')[-1])
                        for idx in range(oplen):
                            tagger[moop['pos'] + idx + 1] = 'M'
                            mask_label[moop['pos'] + idx + 1] = moop['label_token'][idx]
                        for idx in range(opdel):
                            tagger[moop['pos'] + oplen + idx + 1] = 'MD'
                    elif 'INS' in moop['tag']:
                        oplen = eval(moop['tag'].split('+')[0].split('_')[-1])
                        opins = eval(moop['tag'].split('+')[-1].split('_')[-1])
                        for idx in range(oplen - 1):
                            tagger[moop['pos'] + idx + 1] = 'M'
                            mask_label[moop['pos'] + idx + 1] = moop['label_token'][idx]
                        tagger[moop['pos'] + oplen] = 'MI'
                        mod_label[moop['pos'] + oplen] = opins
                        mask_label[moop['pos'] + oplen] = moop['label_token'][oplen - 1:]
                    else:
                        raise Exception("Operator contains unkown tag %s" % moop['tag'])
        return tagger, mask_label, ins_label, mod_label

    def __repr__(self):
        obreturn = ''
        obreturn += ('>>> TaggerConverter Elements : ' + '\n')
        if len(self.labels) == 0 or len(self.tagger_sequence) == 0:
            obreturn += 'Null object'
            return obreturn
        obreturn += 'Original sentence : {}\n'.format(self.origin_sentence)
        obreturn += ('> Tagger sequence :\n')
        obreturn += (', '.join([ele for ele in self.tagger_sequence]) + '\n')
        obreturn += ('> convert sentence :' + '\n')
        obreturn += (''.join([ele for ele in self.post_tokens]) + '\n')
        obreturn += ('> Mask label :\n')
        obreturn += (', '.join(['({}, {})'.format(ele, self.mask_label[ele]) for ele in self.mask_label.keys()]) + '\n')
        obreturn += ('> Insert label :\n')
        obreturn += (', '.join(['{}'.format(ele, self.ins_label[ele]) for ele in self.ins_label]) + '\n')
        obreturn += ('> Modify label :\n')
        obreturn += (', '.join(['{}'.format(ele, self.mod_label[ele]) for ele in self.mod_label]) + '\n')
        return obreturn