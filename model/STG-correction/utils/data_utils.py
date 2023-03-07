import json

import numpy as np
import pandas as pd
from argparse import Namespace

import xlsxwriter

from utils.model_op import _padding as padding
from utils.defines import TAGGER_MAP, DELETE_TAG, MODIFY_DELETE_TAG, MOIFY_ONLY_TAG, MODIFY_TAG, INSERT_TAG, KEEP_TAG, MASK_LM_ID, TYPE_MAP_INV, TYPE_MAP_INV_NEW
del_ops = {}

def combine_insert_modify(insert_label, modify_label):
    comb_label = []
    assert len(insert_label) == len(modify_label)
    size = len(insert_label)
    for idx in range(size):
        if insert_label[idx] != -1 and modify_label[idx] != -1: raise Exception("Error combinition.")
        elif insert_label[idx] != -1:
            comb_label.append(insert_label[idx])
        elif modify_label[idx] != -1:
            comb_label.append(modify_label[idx])
        else: comb_label.append(-1)
    return comb_label

def compare_iner_operate(sentence : str, ops : list) -> dict:
    op_num = []
    def iner_same_compare(sentence : str, opis : list) -> dict:
        oper_num = []
        for op in opis:
            tmp_opnum = 0
            if 'Delete' in op.keys(): tmp_opnum += len(op['Delete'])
            if 'Insert' in op.keys():
                for ins in op['Insert']: tmp_opnum += len(ins['label'])
            if 'Modify' in op.keys():
                for ins in op['Modify']: tmp_opnum += len(ins['label'])
            oper_num.append(tmp_opnum)
        argmin = oper_num.index(min(oper_num))
        if 0 in [oele - min(oper_num) for oele in oper_num]:
            mini_ops = [opis[i] for i, x in enumerate(oper_num) if x == min(oper_num)]
            return mini_ops[0]
        else: return opis[argmin]
    for opidx in range(len(ops)):
        op = ops[opidx]
        op_num.append(len(op.keys()))
    argmin = op_num.index(min(op_num))
    if 0 in [oele - min(op_num) for oele in op_num]:
        min_ops = [ops[i] for i, x in enumerate(op_num) if x == min(op_num)]
        return iner_same_compare(sentence, min_ops)
    else: return ops[argmin]

def data_filter(sentences : list, operates : list):
    filter_operates = []
    print('>>> Select Operate Mode')
    for oidx in range(len(operates)):
        operate = operates[oidx]
        sentence = sentences[oidx]
        if len(operate) < 1: filter_operates.append({})
        elif len(operate) < 2: filter_operates.append(operate[0])
        elif len(operate) > 1: filter_operates.append(compare_iner_operate(sentence, operate))
    return filter_operates

def extract_generate_tokens(labels, tokenizer):
    tgt_gens = []
    for idx in range(len(labels)):
        label = labels[idx]
        gen_dict, gen = {}, []
        if 'Insert' in label.keys():
            for ins in label['Insert']:
                gen_dict[ins['pos']] = ins['label_token']
        if 'Modify' in label.keys():
            for mod in label['Modify']:
                gen_dict[mod['pos']] = mod['label_token']
        if len(gen_dict.keys()) == 0:
            tgt_gens.append([])
            continue
        sort_key = list(gen_dict.keys())
        sort_key.sort()
        for key in sort_key:
            gen.extend(gen_dict[key])
        tgt_gens.append(tokenizer.convert_tokens_to_ids(gen))
    return tgt_gens

def reconstruct_switch(ori_tokens : np.array, switch_preds : np.array, wopad : bool = False):
    post_tokens = []
    batch_size, seq_len = ori_tokens.shape
    for lidx in range(batch_size):
        ori_token = ori_tokens[lidx]
        post_token = [101]
        switch_pred = switch_preds[lidx]
        sw_pidx = switch_pred[0]
        while sw_pidx not in [0, -1] :
            post_token.append(ori_token[sw_pidx])
            sw_pidx = switch_pred[sw_pidx]
            if ori_token[sw_pidx] == 102: switch_pred[sw_pidx] = 0
        assert len(post_token) == np.sum(ori_token > 0)
        post_tokens.append(post_token)
    if wopad is not True:
        return padding(post_tokens, seq_len, 0)
    else:
        return post_tokens, padding(post_tokens, seq_len, 0)

def reconstruct_tagger(tag_tokens : np.array, tag_preds : tuple) -> tuple:
    post_tokens, mlm_tgt_masks = [], []
    tagger, insert, modify = tag_preds
    batch_size, seq_len = tag_tokens.shape
    for lidx in range(batch_size):
        post_token, mlm_mask = [], []
        tag_cur = tagger[lidx]
        ins_cur = insert[lidx]
        mod_cur = modify[lidx]
        token_cur = tag_tokens[lidx]
        for cidx in range(seq_len):
            if tag_cur[cidx] == TAGGER_MAP['PAD']: break   # Pad ignore
            elif tag_cur[cidx] == TAGGER_MAP[KEEP_TAG]:
                mlm_mask.append(0)
                post_token.append(token_cur[cidx])
            elif tag_cur[cidx] in [TAGGER_MAP[DELETE_TAG], TAGGER_MAP[MODIFY_DELETE_TAG]]:
                continue
            elif tag_cur[cidx] == TAGGER_MAP[INSERT_TAG]:
                insert_num = ins_cur[cidx]
                if (insert_num < 1): continue
                post_token.append(token_cur[cidx])
                mlm_mask.append(0)
                post_token.extend([MASK_LM_ID] * insert_num)
                mlm_mask.extend([1] * insert_num)
            elif tag_cur[cidx] == TAGGER_MAP[MOIFY_ONLY_TAG]:
                mlm_mask.append(1)
                post_token.append(MASK_LM_ID)
            elif tag_cur[cidx] == TAGGER_MAP[MODIFY_TAG]:
                modify_num = mod_cur[cidx]
                post_token.append(MASK_LM_ID)
                mlm_mask.append(1)
                if (modify_num < 1): continue
                post_token.extend([MASK_LM_ID] * modify_num)
                mlm_mask.extend([1] * modify_num)
        post_tokens.append(post_token)
        mlm_tgt_masks.append(mlm_mask)
    return post_tokens, mlm_tgt_masks

def reconstruct_tagger_V2(tag_tokens : np.array, tag_preds : tuple, return_flag : bool = False) -> tuple:
    post_tokens, mlm_tgt_masks, op_flag, sp_mapper = [], [], [], []
    tagger, insmod = tag_preds
    batch_size, seq_len = tag_tokens.shape
    for lidx in range(batch_size):
        post_token, mlm_mask = [], []
        tag_cur = tagger[lidx]
        insmod_cur = insmod[lidx]
        token_cur = tag_tokens[lidx]
        flag, curidx, mapper = False, -1, {}
        for cidx in range(seq_len):
            if tag_cur[cidx] == TAGGER_MAP['PAD']: break   # Pad ignore
            elif tag_cur[cidx] == TAGGER_MAP[KEEP_TAG]:
                mlm_mask.append(0)
                post_token.append(token_cur[cidx])
                curidx += 1
                mapper[cidx] = curidx
            elif tag_cur[cidx] in [TAGGER_MAP[DELETE_TAG], TAGGER_MAP[MODIFY_DELETE_TAG]]:
                flag = True
                mapper[cidx] = -1
                continue
            elif tag_cur[cidx] == TAGGER_MAP[INSERT_TAG]:
                flag = True
                insert_num = insmod_cur[cidx]
                if (insert_num < 1): continue
                post_token.append(token_cur[cidx])
                mlm_mask.append(0)
                post_token.extend([MASK_LM_ID] * insert_num)
                mlm_mask.extend([1] * insert_num)
                curidx += 1
                mapper[cidx] = curidx
                curidx += insert_num
            elif tag_cur[cidx] == TAGGER_MAP[MOIFY_ONLY_TAG]:
                flag = True
                mlm_mask.append(1)
                post_token.append(MASK_LM_ID)
                curidx += 1
                mapper[cidx] = -1
            elif tag_cur[cidx] == TAGGER_MAP[MODIFY_TAG]:
                flag = True
                modify_num = insmod_cur[cidx]
                post_token.append(MASK_LM_ID)
                mlm_mask.append(1)
                if (modify_num < 1): continue
                post_token.extend([MASK_LM_ID] * modify_num)
                mlm_mask.extend([1] * modify_num)
                mapper[cidx] = -1
                curidx += (modify_num + 1)
        post_tokens.append(post_token)
        mlm_tgt_masks.append(mlm_mask)
        op_flag.append(flag)
        sp_mapper.append(mapper)
    if return_flag:
        return post_tokens, mlm_tgt_masks, sp_mapper, op_flag
    else:
        return post_tokens, mlm_tgt_masks, sp_mapper

def map_unk2word(tokens : list, sentence : str):
    map_dict, sentid = {}, 0
    for idx in range(1, len(tokens)-1):
        if sentence[sentid] == tokens[idx]:
            sentid += 1
        else:
            if tokens[idx].startswith('#'):
                tmptok = tokens[idx].replace('#', '')
                sentid += len(tmptok)
            elif tokens[idx] == '[UNK]':
                map_dict[idx] = sentence[sentid]
                sentid+=1

    return map_dict

def fillin_tokens4gts(generator_tokens, mlm_tgts):
    size = len(generator_tokens)
    data_out = []
    for lidx in range(size):
        tokens = generator_tokens[lidx]
        tgts = mlm_tgts[lidx]
        if len(tgts) < 1:
            data_out.append(tokens)
            continue
        counter, token = 0, []
        for idx in range(len(tokens)):
            if tokens[idx] == 103:
                try:
                    token.append(tgts[counter])
                except:
                    print(token, counter)
                counter += 1
            else:
                token.append(tokens[idx])
        data_out.append(token)
    return data_out



def fillin_tokens(generator_tokens, mlm_masks, mlm_tgts):
    size = len(generator_tokens)
    data_out, tgt_counter = [], 0
    for lidx in range(size):
        tokens = generator_tokens[lidx]
        masks = mlm_masks[lidx]
        posts = []
        length = len(tokens)
        assert length == len(masks)
        for idx in range(length):
            if masks[idx] == 1 and tokens[idx] == 103:
                posts.append(mlm_tgts[tgt_counter])
                tgt_counter += 1
            elif (masks[idx] == 1 and tokens[idx] != 103) and (masks[idx] == 0 and tokens[idx] == 103):
                raise Exception('Error Instance.')
            else:
                 posts.append(tokens[idx])
        data_out.append(posts)
    return data_out

def deparse_generator(genertae_tokens : list, genertae_label : tuple):
    _, mlm_preds, mlm_masks = genertae_label
    mlm_labels = []
    gidx = 0
    for tid in range(len(genertae_tokens)):
        gen_token = genertae_tokens[tid]
        gnum = len(np.where(np.array(gen_token) == 103)[0])
        mlm_label = mlm_preds[gidx: gidx+gnum]
        gidx += gnum
        mlm_labels.append(mlm_label)
    return mlm_labels

def map_unk2word_inv(tokens : list, map_unk : dict):
    for unk_id in map_unk.keys():
        if unk_id-1 < len(tokens) and tokens[unk_id-1] == '[UNK]':
            tokens[unk_id-1] = map_unk[unk_id]
    return tokens

def output_switch_analysis(args, testdata, pred_tokens, gts_tokens):
    import operator as op
    size = len(testdata)
    out_data = []
    for idx in range(size):
        sentence = testdata.sentences[idx]
        flag = False
        operate = testdata.origin_label[idx]
        for ops in operate:
            for key in ops.keys():
                if key == 'Switch': flag=True
        pred_sentence = ''.join(testdata.tokenizer.convert_ids_to_tokens(pred_tokens[idx])[1:-1])
        label = 1 if op.eq(pred_tokens[idx], gts_tokens[idx]) else 0
        flag = 1 if flag else 0
        out_data.append([sentence, pred_sentence,flag,  label])
    return pd.DataFrame(np.array(out_data), columns=['Sentence', 'Preds', 'Switch', 'Label'])

def obtain_uuid(file_path : str):
    with open(file_path, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    fp.close()
    return list(test_data.keys())

def joint_report(args, file_path : str, res_info : list, testset):
    binary_res, types_res, switch_res, tagger_res, gen_res = res_info
    binary_preds, binary_gts = binary_res
    type_preds, type_gts = types_res
    switch_gts, switch_preds, _ = switch_res
    tagger_gts, tagger_preds, _ = tagger_res
    original_sentence, ori_tokens = testset.sentences, testset.wd_idx
    switch_token_preds  = reconstruct_switch(padding(ori_tokens, args.padding_size, args.padding_val), np.array(switch_preds))
    switch_token_gts    = reconstruct_switch(padding(ori_tokens, args.padding_size, args.padding_val), np.array(switch_gts))
    tagger_token_preds  = reconstruct_tagger(switch_token_preds, tagger_preds)
    tagger_token_gts    = reconstruct_tagger(switch_token_gts, tagger_gts)
    generate_tokens     = deparse_generator(tagger_token_preds[0], gen_res)
    operates_json = testset.operates
    test_size = len(testset)
    results = []
    for idx in range(test_size):
        sentence = original_sentence[idx]
        binary_pred = binary_preds[idx]
        operate = operates_json[idx]
        binary_gt = binary_gts[idx]
        binary_flag = 'TRUE' if binary_gt == binary_pred else 'FALSE'
        type_pred = ','.join([TYPE_MAP_INV[i] for i,x in enumerate(type_preds[idx].tolist()) if x!=0])
        type_gt = ','.join([TYPE_MAP_INV[i] for i,x in enumerate(type_gts[idx].tolist()) if x!=0])
        type_flag = 'TRUE' if type_pred == type_gt else 'FALSE'
        switch_pred_token, switch_gts_token = switch_token_preds[idx][1:], switch_token_gts[idx][1:]
        switch_pred = ''.join(map_unk2word_inv(testset.tokenizer.convert_ids_to_tokens(switch_pred_token), testset.unk_map[idx])).replace('[PAD]', '').replace('[SEP]', '').replace('##', '')
        switch_gt = ''.join(map_unk2word_inv(testset.tokenizer.convert_ids_to_tokens(switch_gts_token), testset.unk_map[idx])).replace('[PAD]', '').replace('[SEP]', '').replace('##', '')
        switch_flag = 'TRUE' if np.max(np.diff(switch_preds[idx])) > 1 else 'FALSE'
        switch_label = 'TRUE' if switch_pred == switch_gt else 'FALSE'
        need_switch = 'TRUE' if np.max(np.diff(switch_gts[idx])) > 1 else 'FALSE'
        tagger_pred_token, tagger_gt_token = tagger_token_preds[0][idx][1:], tagger_token_gts[0][idx][1:]
        tagger_pred = ''.join(map_unk2word_inv(testset.tokenizer.convert_ids_to_tokens(tagger_pred_token), testset.unk_map[idx])).replace('[PAD]', '').replace('[SEP]', '').replace('##', '')
        tagger_gt   = ''.join(map_unk2word_inv(testset.tokenizer.convert_ids_to_tokens(tagger_gt_token), testset.unk_map[idx])).replace('[PAD]', '').replace('[SEP]', '').replace('##', '')
        tagger_flag = 'TRUE' if tagger_pred == tagger_gt else 'FALSE'
        generate_token = ''.join(map_unk2word_inv(testset.tokenizer.convert_ids_to_tokens(generate_tokens[idx]), testset.unk_map[idx])) if len(generate_tokens[idx]) > 0 else ' '
        results.append([sentence, binary_pred, binary_gt, binary_flag, type_pred, type_gt, type_flag, switch_pred, switch_gt, switch_flag, switch_label, need_switch, tagger_pred, generate_token, tagger_gt, tagger_flag, operate])
    df = pd.DataFrame(np.array(results), columns=['Sentence', 'Binary_Pred', 'Binary_Gt', 'Binary_Flag', 'Type_Pred', 'Type_Gt', 'Type_Flag', 'Switch_Pred', 'Switch_Gt', 'Switch_Flag', 'Switch_Label', 'need_switch', 'Tagger_Pred', 'Generate_Pred', 'Tagger_Gt', 'Tagger_Flag', 'Operate'])
    df.to_csv(file_path, index=False, encoding='utf_8_sig')

def report_pipeline_output(out_path, sentences, labels, switchs, taggers, outputs, tgt_tokens = None, eq_label = None, uuid = None):
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet('data')
    if tgt_tokens is not None:
        header = ['Sentence', 'Label', 'Switch', 'Tagger', 'Output', 'tgt', 'Eq']
    else:
        header = ['Sentence', 'Label', 'Switch', 'Tagger', 'Output']
    if uuid is not None:
        header = ['UUID'] + header
    worksheet.write_row(0, 0, header)
    row_id = 1
    size = len(sentences)
    for idx in range(size):
        sentence = sentences[idx]
        label = labels[idx]
        switch = switchs[idx]
        tagger = taggers[idx]
        output = outputs[idx]
        eqlab  = eq_label[idx] if eq_label is not None else None
        tgt = tgt_tokens[idx] if tgt_tokens is not None else None
        collection = [sentence, json.dumps(label, ensure_ascii=False), switch, tagger, output, tgt, eqlab] if tgt_tokens is not None else [sentence, json.dumps(label, ensure_ascii=False), switch, tagger, output]
        if uuid is not None:
            collection = [uuid[idx]] + collection
        worksheet.write_row(row_id, 0, collection)
        row_id += 1
    workbook.close()

def output_type_report(out_path : str, testset, result):
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet('data')
    header = ['Sentence', 'Pred', 'Gts', 'Eq']
    worksheet.write_row(0, 0, header)
    type_preds, type_gts = result
    cal_error, cal_total = {}, {}
    confusion_matrix = np.zeros((len(TYPE_MAP_INV), len(TYPE_MAP_INV)))
    for key in TYPE_MAP_INV_NEW.keys():
        cal_error[TYPE_MAP_INV_NEW[key]] = 0
        cal_total[TYPE_MAP_INV_NEW[key]] = 0
    for idx in range(len(testset)):
        type_pred = ','.join([TYPE_MAP_INV[i] for i, x in enumerate(type_preds[idx].tolist()) if x != 0])
        type_gt = ','.join([TYPE_MAP_INV[i] for i, x in enumerate(type_gts[idx].tolist()) if x != 0])
        type_flag = type_pred == type_gt
        type_gts_cal = [TYPE_MAP_INV_NEW[TYPE_MAP_INV[i]] for i, x in enumerate(type_gts[idx].tolist()) if x != 0]
        for typ in type_gts_cal:
            cal_total[typ] += 1
            pred_typ = [i for i, x in enumerate(type_preds[idx].tolist()) if x != 0]
            if len(pred_typ) < 1: continue
            else: pred_typ = pred_typ[0]
            gts_typ = [i for i, x in enumerate(type_gts[idx].tolist()) if x != 0]
            confusion_matrix[pred_typ][gts_typ] += 1
        if type_flag is not True:
            for typ in type_gts_cal:
                cal_error[typ] += 1
        sentence = testset.sentences[idx]
        worksheet.write_row(idx+1, 0, [sentence, type_pred, type_gt, type_flag])
    print('>> Error Case Count:')
    print(cal_error)
    print('>> Total Case Count:')
    print(cal_total)
    workbook.close()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot = sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, fmt='0.20g', square=True)
    axis_label = [TYPE_MAP_INV_NEW[ele] for ele in TYPE_MAP_INV_NEW.keys()]
    ax.set_xticklabels(axis_label, font='Times New Roman', fontsize=15)
    ax.set_yticklabels(axis_label, font='Times New Roman', fontsize=15)
    plt.savefig('dataset/EMNLP/Multiple/multiple_confusion.pdf', dpi=500)
    plt.show()

def convert_spmap_sw(sp_map : list, sw_labels : np.array) -> list:
    new_spmap = []
    for idx in range(len(sp_map)):
        mapper = sp_map[idx]
        if isinstance(mapper, bool): new_spmap.append(None)
        else:
            swlabel = sw_labels[idx]
            sw_pidx, pt = swlabel[0], 0
            t_idx, map_trans = 0, {}
            while sw_pidx not in [0, -1]:
                map_trans[sw_pidx - 1] = pt
                sw_pidx = swlabel[sw_pidx]
                pt += 1
            spmap = dict([[map_trans[ele], mapper[ele]] for ele in mapper if ele in map_trans])
            new_spmap.append(spmap)
    return new_spmap

def convert_spmap_tg(sp_map : list, tg_mapper : list) -> list:
    new_spmap = []
    for idx in range(len(sp_map)):
        if idx == 163:
            print()
        spmapper = sp_map[idx]
        tgmapper = tg_mapper[idx]
        mapper = {}
        if not spmapper: new_spmap.append(None)
        else:
            for sp in spmapper:
                if sp in tgmapper and tgmapper[sp] >= 0: mapper[tgmapper[sp]] = spmapper[sp]
            new_spmap.append(mapper)
    return new_spmap

def convert_spmap2tokens(tokens : list, sp_maps : dict):
    if not sp_maps: return tokens
    for i,x in enumerate(tokens):
        if i in sp_maps: tokens[i] = sp_maps[i]
    return tokens