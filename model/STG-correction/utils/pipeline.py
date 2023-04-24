import time
from .defines import SPLIT_VOCAB, INNER_VOCAB
from math import ceil

def base_context(report_title :str, document : str, switch :bool = True, modify : bool = True) -> dict:
    context = {}
    time_str = ''.join([time.asctime(time.localtime(time.time()))])
    context['report_name'] = report_title
    context['report_time'] = time_str
    context['process_switch'] = '√' if switch else '×'
    context['process_modify'] = '√' if modify else '×'
    context['origin_text'] = document
    return context

def split_sentence(document : str, padding_size : int = 150, min_len : int = 10) -> list:
    '''
    Split Sentence to small piece (\n, 。, ？ ,etc.)
    :param document:
    :param padding_size:
    :return: small piece sentences [list]
    '''
    sentences = []
    split_candidates = [i for i, x in enumerate(document) if x in SPLIT_VOCAB]
    split_candidates = [-1] + split_candidates if 0 not in split_candidates else split_candidates
    split_candidates = split_candidates + [len(split_candidates)-1]  if len(split_candidates) - 1 else split_candidates
    buffer = ''
    for sidx in range(1, len(split_candidates)):
        buffer += document[split_candidates[sidx-1] + 1 : split_candidates[sidx]+1]
        if len(buffer) < padding_size - 3 and len(buffer) > min_len:
            sentences.append(buffer)
            buffer = ''
        else:
            if len(buffer) < min_len:
                continue
            elif len(buffer) >= padding_size - 3:
                inner_candidate =  [i for i, x in enumerate(buffer) if x in INNER_VOCAB]
                if len(inner_candidate) > 0:
                    inner_pos = inner_candidate[ceil(len(inner_candidate)/2) - 1]
                    tmp_sentence = [buffer[:inner_candidate[inner_pos]], buffer[inner_candidate[inner_pos]:]]
                    sentences.extend(tmp_sentence)
                    buffer = ''
                else:
                    tmp_sentence = [buffer[:ceil(len(buffer)/2)], buffer[ceil(len(buffer)/2):]]
                    sentences.extend(tmp_sentence)
                    buffer = ''
    return sentences