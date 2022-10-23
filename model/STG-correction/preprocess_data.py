import sys
sys.path.append('')
import argparse
from utils.argument import ArgumentGroup
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def args_parse():
    parser = argparse.ArgumentParser(description='FCGEC preprocess params')
    base_args = ArgumentGroup(parser, 'base', 'Base Settings')

    base_args.add_arg('mode', str, 'normal', 'STG Mode')
    base_args.add_arg('err_only', bool, True, 'Construct error dataset')
    base_args.add_arg('data_dir', str, 'dataset', 'Dataset path')
    base_args.add_arg('out_dir', str, 'STG-Indep', 'Output path')
    base_args.add_arg('train_file', str, '', 'Train path')
    base_args.add_arg('valid_file', str, '', 'Valid path')
    base_args.add_arg('test_file', str, '', 'Test path')

    args = parser.parse_args()
    return args

def process_train_valid(path, desc = 'train', err_only = True, need_type = False) -> pd.DataFrame:
    print('[TASK] processing {} file.'.format(desc))
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    fp.close()
    corrects = []
    for key in tqdm(data, 'Processing'):
        element = data[key]
        error_type = element['error_type']
        sentence = element['sentence']
        operate = element['operation']
        if err_only:
            if error_type != '*': corrects.append([sentence, error_type, operate]) if need_type else corrects.append([sentence, operate])
        else:
            corrects.append([sentence, error_type, operate]) if need_type else corrects.append([sentence, operate])
    if need_type:
        return pd.DataFrame(corrects, columns=['Sentence', 'Type', 'Label'])
    else:
        return pd.DataFrame(corrects, columns=['Sentence', 'Label'])

def process_test(path) -> pd.DataFrame:
    print('[TASK] processing test file.')
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    fp.close()
    corrects = []
    for key in tqdm(data, 'Processing'):
        element = data[key]
        sentence = element['sentence']
        corrects.append([sentence, "[]"])
    return pd.DataFrame(corrects, columns=['Sentence', 'Label'])

def preprocess_independent(args):
    print('=' * 20 + "Preprocess Data for STG Independent" + "=" * 20)
    assert os.path.join(args.data_dir, args.train_file)
    assert os.path.join(args.data_dir, args.valid_file)
    assert args.train_file != ''
    assert args.valid_file != ''
    if args.test_file != '':
        assert os.path.join(args.data_dir, args.test_file)
    if not os.path.exists(os.path.join(args.data_dir, args.out_dir)):
        os.mkdir(os.path.join(args.data_dir, args.out_dir))
    # Process Train
    train_df = process_train_valid(os.path.join(args.data_dir, args.train_file), 'train', err_only=args.err_only)
    train_df.to_csv(os.path.join(args.data_dir, args.out_dir, 'train.csv'), index=False, encoding='utf_8_sig')
    print('Processed train dataset, saved at %s' % os.path.join(os.path.join(args.data_dir, args.out_dir, 'train.csv')))
    # Process Valid
    valid_df = process_train_valid(os.path.join(args.data_dir, args.valid_file), 'valid', err_only=args.err_only)
    valid_df.to_csv(os.path.join(args.data_dir, args.out_dir, 'valid.csv'), index=False, encoding='utf_8_sig')
    print('Processed valid dataset, saved at %s' % os.path.join(os.path.join(args.data_dir, args.out_dir, 'valid.csv')))
    if args.test_file != '':
        test_df = process_test(os.path.join(args.data_dir, args.test_file))
        test_df.to_csv(os.path.join(args.data_dir, args.out_dir, 'test.csv'), index=False, encoding='utf_8_sig')
        print('Processed test dataset, saved at %s' % os.path.join(os.path.join(args.data_dir, args.out_dir, 'test.csv')))

def preprocess_independent_TTI(args):
    print('=' * 20 + "Preprocess Data for STG Independent+TTI" + "=" * 20)
    assert os.path.join(args.data_dir, args.train_file)
    assert os.path.join(args.data_dir, args.valid_file)
    assert args.train_file != ''
    assert args.valid_file != ''
    if args.test_file != '':
        assert os.path.join(args.data_dir, args.test_file)
    if not os.path.exists(os.path.join(args.data_dir, args.out_dir)):
        os.mkdir(os.path.join(args.data_dir, args.out_dir))
    # Process Train
    train_df = process_train_valid(os.path.join(args.data_dir, args.train_file), 'train', err_only=args.err_only, need_type=True)
    train_df.to_csv(os.path.join(args.data_dir, args.out_dir, 'train.csv'), index=False, encoding='utf_8_sig')
    print('Processed train dataset, saved at %s' % os.path.join(os.path.join(args.data_dir, args.out_dir, 'train.csv')))
    # Process Valid
    valid_df = process_train_valid(os.path.join(args.data_dir, args.valid_file), 'valid', err_only=args.err_only, need_type=True)
    valid_df.to_csv(os.path.join(args.data_dir, args.out_dir, 'valid.csv'), index=False, encoding='utf_8_sig')
    print('Processed valid dataset, saved at %s' % os.path.join(os.path.join(args.data_dir, args.out_dir, 'valid.csv')))
    if args.test_file != '':
        test_df = process_test(os.path.join(args.data_dir, args.test_file))
        test_df.to_csv(os.path.join(args.data_dir, args.out_dir, 'test.csv'), index=False, encoding='utf_8_sig')
        print('Processed test dataset, saved at %s' % os.path.join(os.path.join(args.data_dir, args.out_dir, 'test.csv')))



if __name__ == '__main__':
    args = args_parse()
    if args.mode == 'normal':
        preprocess_independent(args)
    elif args.mode == 'tti':
        preprocess_independent_TTI(args)