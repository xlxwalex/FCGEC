import torch
from utils.defines import MASK_LOGIT_CONST, KEEP_TAG, MODIFY_TAG, INSERT_TAG, GEN_KEEP_LABEL, DELETE_TAG, MOIFY_ONLY_TAG, MODIFY_DELETE_TAG, MASK_SYMBOL

def logits_mask(inputs :torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
    mask_add =  -MASK_LOGIT_CONST * (1. - mask)
    scores = inputs * mask + mask_add
    return scores

class SelfAttentionMask(object):
    '''
    Create Attention Mask From 2-D Mask Tensor
    '''
    def __call__(self, inputs : torch.Tensor, mask : torch.Tensor = None, diag_mask : bool = True, need_mask : bool = False) -> torch.Tensor:
        '''
        Create 3D Tensor of Mask For PointerNetwork
        :param inputs: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        :param mask: int32 Tensor of shape [batch_size, to_seq_length]
        :param diag_mask: whether mask diag (To prevent self-loop)
        :return: float Tensor of shape [batch_size, seq_length, seq_length].
        '''
        if isinstance(inputs, list) and mask is None:
            mask = inputs[1]
            inputs = inputs[0]
        mask = mask.type_as(inputs)
        # BroadCast
        mask = mask.unsqueeze(1) * mask.unsqueeze(-1)
        # Diagonal Mask Operator
        if diag_mask:
            diag = torch.cat([torch.diag_embed(torch.diag(ins)).unsqueeze(0) for ins in mask], dim=0)
            mask = mask - diag
        return mask

def convert_tagger2generator(tokens : list, tagger : list, mask_label : dict) -> tuple:
    '''
    Convert Tag (I / MI) To Target Sequence with [Mask] Symbol
    :param tokens: tokens from tokenizer (list)
    :param tagger: tag_sequence list (From TaggerConvertor Label)
    :param mask_label: mask tgt labels (list) from tokenizer
    :return: tokens (list), tgt_labels (list)
    '''
    post_sequence, post_label = [], []
    for index in range(len(tagger)):
        if tagger[index] == KEEP_TAG:
            post_sequence.append(tokens[index])
            post_label.append(GEN_KEEP_LABEL)
        elif tagger[index] in [DELETE_TAG, MODIFY_DELETE_TAG]:
            continue
        elif tagger[index] == INSERT_TAG:
            post_sequence.append(tokens[index])
            post_label.append(GEN_KEEP_LABEL)
            assert index in mask_label.keys()
            post_sequence.extend([MASK_SYMBOL] * len(mask_label[index]))
            post_label.extend(mask_label[index])
        elif tagger[index] == MODIFY_TAG:
            assert index in mask_label.keys()
            post_sequence.extend([MASK_SYMBOL] * len(mask_label[index]))
            post_label.extend(mask_label[index])
        elif tagger[index] == MOIFY_ONLY_TAG:
            post_sequence.append(MASK_SYMBOL)
            assert index in mask_label.keys() and isinstance(mask_label[index], int)
            post_label.append(mask_label[index])
    return post_sequence, post_label
