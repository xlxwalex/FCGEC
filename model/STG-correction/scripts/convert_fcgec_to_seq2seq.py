import sys
sys.path.append('..')
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
from utils.argument import ArgumentGroup
from copy import deepcopy

def args_parse():
    parser = argparse.ArgumentParser(description='FCGEC seq2seq convertor params')
    base_args = ArgumentGroup(parser, 'converter', 'Base Settings')

    base_args.add_arg('out_uuid', bool, True, 'Output UUID in files')
    base_args.add_arg('data_dir', str, '../dataset/', 'Dataset path')
    base_args.add_arg('out_dir', str, '../dataset/', 'Output data path')
    base_args.add_arg('train_file', str, 'FCGEC_train.json', 'Train data path')
    base_args.add_arg('valid_file', str, 'FCGEC_valid.json', 'Valid data path')
    base_args.add_arg('test_file', str, 'FCGEC_test.json', 'Test data path')

    base_args.add_arg('out_errflag', bool, True, 'Whether to output `error_flag`')
    base_args.add_arg('out_errtype', bool, True, 'Whether to output `error_type`')

    args = parser.parse_args()
    return args


def read_json_data(args : argparse.Namespace) -> tuple:
    train_data, valid_data, test_data = None, None, None
    def read_data(path :str) -> list:
        if os.path.exists(path) is not True: return None
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        fp.close()
        return data
    if args.train_file != '': train_data = read_data(os.path.join(args.data_dir, args.train_file))
    if args.valid_file != '': valid_data = read_data(os.path.join(args.data_dir, args.valid_file))
    if args.test_file != '': test_data = read_data(os.path.join(args.data_dir, args.test_file))
    return train_data, valid_data, test_data



def train_valid_processor(out_path : str, dataset : dict, uuid : bool=True, out_flag : bool=True, out_type : bool=True, desc='Train'):
    out_path = out_path.replace('.json', '.csv')
    out_data = []

    def convert_operator2seq(sentence : str, operate : list) -> list:

        def unpack(operate : list) -> list:
            operates = []
            version_old_mode = True
            for ops in operate:
                if version_old_mode and ('Insert' in ops.keys() or 'Modify' in ops.keys()):
                    if 'Insert' in ops.keys() and isinstance(ops['Insert'][0]['label'], list): version_old_mode = False
                    if 'Modify' in ops.keys() and isinstance(ops['Modify'][0]['label'], list): version_old_mode = False
                if version_old_mode:
                    operates.append(ops)
                    continue
                unpacks = []
                def convert_lab(opr : dict, label : str) -> dict:
                    op = deepcopy(opr)
                    op['label'] = label
                    return op
                if 'Insert' in ops.keys():
                    unpacks.extend([{'Insert' : [convert_lab(op, lab)]} for op in ops['Insert'] for lab in op['label']])
                if 'Modify' in ops.keys():
                    unpacks.extend([{'Modify' : [convert_lab(op, lab)]} for op in ops['Modify'] for lab in op['label']])
                operates.extend(unpacks)
            return operates

        def get_postsentence(sentence, operate):
            ret = []
            for op in operate:
                if 'Switch' in op.keys():
                    sentence = ''.join(np.array([s for s in sentence])[np.array(op['Switch'])])
                sentag = [[s, 'O', ''] for s in sentence]
                if 'Delete' in op.keys():
                    for i in op['Delete']:
                        sentag[i][1] = 'D'
                if 'Insert' in op.keys():
                    for i in op['Insert']:
                        sentag[i['pos']][1] = i['tag']
                        sentag[i['pos']][-1] = i['label']
                if 'Modify' in op.keys():
                    for i in op['Modify']:
                        sentag[i['pos']][1] = i['tag']
                        sentag[i['pos']][-1] = i['label']
                ret.append(get_psentence(sentag))
            return ret

        def get_psentence(sentag):
            sent = ''
            cou = 0
            for i in range(len(sentag)):
                if i < cou:
                    continue
                if sentag[cou][1] == 'O':
                    sent += sentag[cou][0]
                elif sentag[cou][1] == 'D':
                    cou += 1
                    continue
                elif sentag[cou][1].startswith('INS'):
                    sent += sentag[cou][0]
                    sent += sentag[cou][-1]
                elif sentag[cou][1].startswith('MOD'):
                    modnum = eval(sentag[cou][1].split('+')[0].split('_')[-1])
                    sent += sentag[cou][-1]
                    cou += (modnum - 1)
                cou += 1
            return sent
        operates = unpack(operate)
        return get_postsentence(sentence, operates)

    for datk in tqdm(dataset.keys(), desc='Processing {} data'.format(desc)):
        outs = []
        if uuid: outs.append(datk)
        outs.append(dataset[datk]['sentence'])
        if out_flag: outs.append(dataset[datk]['error_flag'])
        if out_type: outs.append(dataset[datk]['error_type'])
        post_sentences = convert_operator2seq(dataset[datk]['sentence'], json.loads(dataset[datk]['operation'])) if dataset[datk]['error_flag'] == 1 else dataset[datk]['sentence']
        outs.append('\t'.join(post_sentences))
        out_data.append(outs)

    columns = []
    if uuid: columns.append('UUID')
    columns.append('Sentence')
    if out_flag: columns.append('Binary')
    if out_type: columns.append('Type')
    columns.append('Correction')
    df = pd.DataFrame(out_data, columns=columns)
    df.to_csv(out_path, index=False, encoding='utf_8_sig')
    print('%s data has been processed and saved at %s' % (desc, out_path))

def testdata_processor(out_path : str, test_data : dict, uuid : bool=True):
    out_path = out_path.replace('.json', '.csv')
    test_out_data = []
    for datk in tqdm(test_data.keys(), desc='Processing Test data'):
        outs = []
        if uuid: outs.append(datk)
        outs.append(test_data[datk]['sentence'])
        test_out_data.append(outs)
    df = pd.DataFrame(test_out_data, columns=['UUID', 'Sentence'] if uuid else ['Sentence'])
    df.to_csv(out_path, index=False, encoding='utf_8_sig') if uuid else df.to_csv(out_path, index=False, encoding='utf_8_sig')
    print('Test data has been processed and saved at %s' % out_path)

def convert_data2seq(args : argparse.Namespace):
    train_data, valid_data, test_data = read_json_data(args)
    if train_data: train_valid_processor(os.path.join(args.out_dir, args.train_file), train_data, args.out_uuid, args.out_errflag, args.out_errtype)
    if valid_data: train_valid_processor(os.path.join(args.out_dir, args.valid_file), valid_data, args.out_uuid, args.out_errflag, args.out_errtype, 'Valid')
    if test_data: testdata_processor(os.path.join(args.out_dir, args.test_file), test_data, args.out_uuid)

if __name__ == '__main__':
    args = args_parse()
    convert_data2seq(args)
