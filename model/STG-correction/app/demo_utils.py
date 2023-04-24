import random
from .ReportDataset import ReportDataset
import numpy as np
import subprocess
import os
from docxtpl import DocxTemplate
import time
from utils import DELETE_TAG, MODIFY_TAG, MOIFY_ONLY_TAG, KEEP_TAG, INSERT_TAG, TAGGER_MAP, MODIFY_DELETE_TAG, CACHE_DIR, TEMPLATE_FILE_NAME
try:
    import jieba
    JIEBA_FLAG = True
except:
    JIEBA_FLAG = False
    print('jieba are not installed, use default mode.')
BACK_GATHER_NUM = 4
DELETE_SLOTS = ['删除', '删去', '去掉']
INSERT_SLOTS = ['插入', '加上', '补充']
MODIFY_SLOTS = ['改成', '改成', '修改为', '改写为']


def docs2pdf(docPath : str, pdfPath : str, file_name :list =None) -> None:
    """
    convert a doc/docx document to pdf format (linux only, requires libreoffice)
    :param doc: path to document
    """
    cmd = 'libreoffice7.3 --headless --convert-to pdf'.split() + [docPath] + ['--outdir'] + [pdfPath]
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait(timeout=30)
    stdout, stderr = p.communicate()
    if stderr:
        raise subprocess.SubprocessError(stderr)
    if file_name is not None:
        if len(file_name) == 2 and os.path.exists(file_name[0]):
            os.rename(file_name[0], file_name[1])

def context2docs(context : dict, export_path :str = None, doc_type:str='pdf'):
    assert doc_type in ['pdf', 'docx']
    tpl = DocxTemplate(TEMPLATE_FILE_NAME)
    tpl.render(context)
    cache_filename = str(time.time()) + '.docx'
    if export_path is None:
        prefix = './'
    else:
        prefix = export_path
    ex_path = prefix + '_'.join(context['report_name'].split(' ')) + '_' +str(time.time())
    if doc_type == 'pdf':
        ex_path += '.pdf'
        tpl.save(os.path.join(CACHE_DIR, cache_filename))
        docs2pdf(os.path.join(CACHE_DIR, cache_filename), '../', ['../' + cache_filename.replace('.docx', '.pdf'), ex_path])
        os.remove(os.path.join(CACHE_DIR, cache_filename))
    else:
        ex_path += '.docx'
        tpl.save(ex_path)
    return ex_path

def process_switch_inform(tokens, switch_pointer):
    switch_idxs = [0]
    while switch_pointer[switch_idxs[-1]] != 0:switch_idxs.append(switch_pointer[switch_idxs[-1]])
    differ = [i for i in range(1, len(switch_idxs)) if switch_idxs[i] - switch_idxs[i-1] != 1]
    dif_len = len(differ)
    if dif_len < 2: return '句子中成分位置发生改变'
    elif dif_len == 4: return '句子中的"{}"与"{}"应该互换位置'.format(''.join(tokens[differ[0]: differ[1]]), ''.join(tokens[differ[2]: differ[3]]))
    else: return '句子中的"{}"位置应该发生改变'.format(''.join(tokens[differ[0]: differ[1]]))

def process_taggen_inform(tag_token, tag_index, insmods, generate_tokens, gidx):
    inform = []
    D, I, M = [], {}, {}
    offset = 0
    for idx in range(tag_index.shape[0]):
        if tag_index[idx] == 0: break
        if tag_index[idx] == TAGGER_MAP[KEEP_TAG]: continue
        elif tag_index[idx] in [TAGGER_MAP[DELETE_TAG], TAGGER_MAP[MODIFY_DELETE_TAG]]: D.append(idx)
        elif tag_index[idx] == TAGGER_MAP[INSERT_TAG]:
            gen_num = insmods[idx]
            back_token = ''.join(tag_token[max(idx-BACK_GATHER_NUM, 1):idx+1]).replace('[SEP]', '')
            if idx == 0: back_string = '句首'
            else: back_string = jieba.lcut(back_token)[-1] if JIEBA_FLAG else back_token
            I[idx] = [back_string, ''.join(generate_tokens[gidx+offset:gidx+offset+gen_num])]
            offset += gen_num
        elif tag_index[idx] == TAGGER_MAP[MOIFY_ONLY_TAG]:
            M[idx] = [MOIFY_ONLY_TAG, tag_token[idx], ''.join(generate_tokens[gidx + offset:gidx + offset + 1])]
            offset += 1
        elif tag_index[idx] == TAGGER_MAP[MODIFY_TAG]:
            gen_num = insmods[idx] + 1
            M[idx] = [MODIFY_TAG, tag_token[idx], generate_tokens[gidx+offset:gidx+offset+gen_num]]
            offset += gen_num
    # Delete
    temP_D = []
    for idx in D:
        if len(temP_D) == 0: temP_D.append([idx])
        elif idx == temP_D[-1][-1] + 1: temP_D[-1].append(idx)
        else:temP_D.append([idx])
    for comb in temP_D: inform.append('{}"{}"'.format(DELETE_SLOTS[random.randint(0, len(DELETE_SLOTS)-1)], ''.join([tag_token[_] for _ in comb])))
    # Insert
    for ins in I:
        inform.append('在{}{}{}"{}"'.format('"{}"'.format(I[ins][0]) if I[ins][0]!='句首' else '句首', '的后面' if I[ins][0]!='句首' else '', INSERT_SLOTS[random.randint(0, len(INSERT_SLOTS)-1)], I[ins][1]))
    # Modify
    temP_M = []
    for modidx in M:
        if len(temP_M) == 0: temP_M.append([[modidx] + M[modidx]])
        elif modidx == temP_M[-1][-1][0] + 1:
            if temP_M[-1][-1][1] == MODIFY_TAG: temP_M.append([[[modidx] + M[modidx]]])
            else: temP_M[-1].append([modidx] + M[modidx])
        else:temP_M.append([[modidx] + M[modidx]])
    for comb in temP_M:
        inform.append('将"{}"{}"{}"'.format(''.join([_[2] for _ in comb]), MODIFY_SLOTS[random.randint(0, len(MODIFY_SLOTS)-1)], ''.join([_[3] for _ in comb])))
    return inform, offset


def transform2inform(data_bucket: ReportDataset, generate_inform: list, tagger_inform: tuple, switch_inform: tuple):
    tokenizer = data_bucket.tokenizer
    _, switch_tag, switch_pointer = switch_inform
    (tagger_index, insmod_num), _ = tagger_inform
    generate_token = tokenizer.convert_ids_to_tokens(generate_inform)
    informs, global_genidx = [], 0
    for idx in range(len(switch_tag)):
        inform = []
        if switch_tag[idx]:
            ori_token = tokenizer.convert_ids_to_tokens(data_bucket.wd_idx[idx])
            inform.append(process_switch_inform(ori_token, switch_pointer[idx]))
        if np.max(tagger_index[idx]) <= 1:
            if len(inform) == 0: inform.append('该句子没有语病')
        else:
            other_inform, gdelta = process_taggen_inform(tokenizer.convert_ids_to_tokens(data_bucket.wd_tag_idx[idx]), tagger_index[idx], insmod_num[idx], generate_token, global_genidx)
            global_genidx += gdelta
            inform.extend(other_inform)
        informs.append(inform)
    return informs
