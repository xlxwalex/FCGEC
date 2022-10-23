import numpy as np
import torch
from scipy.special import logsumexp

def _padding(inputs : list, paddings : int, pad_val : int) -> np.ndarray:
    doc = np.array([
        np.pad(x[0:paddings], ( 0, paddings - len(x[0:paddings])),
               'constant', constant_values=pad_val)
        for x in inputs
    ]).astype('int64')
    return doc

def _attention_mask(padded : np.ndarray, pad_val : int) -> torch.Tensor:
    np_mask = (padded != pad_val).astype('int32')
    return torch.from_numpy(np_mask)

def _save_model(path : str, checkp : dict) -> None:
    torch.save(checkp, path)

def _normalize_logits(logits):
    numerator = logits
    denominator = logsumexp(logits)
    return numerator - denominator

def _softmax_logits(logits :torch.Tensor, dim : int = 1):
    return torch.softmax(logits, dim=dim)

def _clip_max_generate(gts :torch.Tensor, maxgen : int, sub_num : int = -2):
    if sub_num < -1:
        gts[gts >maxgen] = maxgen
    else:
        gts[gts > maxgen] = sub_num
    return gts

def _generate_tagger_loss_weights(number_classes : int, special_class : list, special_weights : float = 20.) -> torch.Tensor:
    weights = [1.] * number_classes
    for spc in range(len(special_class)):
        weights[spc] = special_weights
    return torch.from_numpy(np.array(weights)).float()