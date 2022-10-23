import pandas as pd
import numpy as np
import xlsxwriter
import json
from utils.defines import MASK_SYMBOL
from tqdm import tqdm

def _fill_mask(tokens : list, mask_ids : list, pred_label : list) -> str:
    '''
    Fill Mask Symbol In Token Sequences with Pred Tokens
    :param tokens: Token sequences (list)
    :param mask_ids: mask index (list)
    :param pred_label: pred token label (list)
    :return: post_sentence (str)
    '''
    assert len(mask_ids) == len(pred_label)
    for idx in range(len(mask_ids)):
        tokens[mask_ids[idx]] = pred_label[idx]
    return  ''.join(tokens)

def export_generator(path : str, dataset, preds, gts, masks):
    sentences, operators, mask_sequences, predicts, labels, post_sentences = [], [], [], [], [], []
    mask = np.clip(np.array(masks), 0, 1)
    mask_length = np.sum(mask, axis=1)
    mask_index  = 0
    pred_tokens = dataset.tokenizer.convert_ids_to_tokens(preds)
    gts_tokens = dataset.tokenizer.convert_ids_to_tokens(gts)
    for index in tqdm(range(len(dataset)), desc='Exporting'):
        sentences.append(dataset.sentences[index])
        operators.append(json.dumps(dataset.labels[index], ensure_ascii=False))
        mask_sequences.append(''.join(dataset.token[index]))
        predicts.append(''.join(pred_tokens[mask_index:mask_index + mask_length[index]]))
        labels.append(''.join(gts_tokens[mask_index:mask_index + mask_length[index]]))
        mask_pos = [i for i, x in enumerate(dataset.token[index]) if x == MASK_SYMBOL]
        post_sentences.append(_fill_mask(dataset.token[index], mask_pos, pred_tokens[mask_index:mask_index + mask_length[index]]))
        mask_index += mask_length[index]
    sentences = np.array(sentences).reshape(len(dataset), 1)
    operators = np.array(operators).reshape(len(dataset), 1)
    mask_sequences = np.array(mask_sequences).reshape(len(dataset), 1)
    post_sentences = np.array(post_sentences).reshape(len(dataset), 1)
    predicts  = np.array(predicts).reshape(len(dataset), 1)
    labels    = np.array(labels).reshape(len(dataset), 1)
    comb_data = np.hstack((sentences, operators, mask_sequences, post_sentences, predicts, labels))
    comb_df   = pd.DataFrame(comb_data, columns=['Sentence', 'Operator', 'Sequence', 'Revision','Prediction', 'Label'])
    comb_df.to_csv(path, index=False, encoding='utf_8_sig')