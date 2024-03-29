from utils.argument import ArgumentGroup
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FCGEC Evaluation Params')

    # Base Params
    base_args = ArgumentGroup(parser, 'base', 'Base Settings')
    base_args.add_arg('mode', str, 'test', 'Experiment Mode')
    base_args.add_arg('cuda', bool, True, 'device : True - CUDA, False - CPU (Force)')
    base_args.add_arg('gpu_id', int, 0, 'GPU Device ID, defaut->cuda:0')
    base_args.add_arg('seed', int, 2022, 'Experiment Seed')
    base_args.add_arg('checkpoints', str, 'checkpoints/', 'Checkpoint Path Dir')
    base_args.add_arg('checkp', str, 'joint_model/', 'Checkpoint Sub Dir)')

    base_args.add_arg('export', str, 'stg_joint_test.xlsx', 'Export file name')

    # Dataset Params
    data_args = ArgumentGroup(parser, 'dataset', 'Dataset Settings')
    data_args.add_arg('data_base_dir', str, 'dataset/stg_joint/', 'Base Dir Of Dataset')

    # Pretrained Model Params
    pretrained_args = ArgumentGroup(parser, 'pretrained', 'Pretrained Model Settings')
    pretrained_args.add_arg('use_lm', bool, True, 'Whether Model Use Language Models')
    pretrained_args.add_arg('lm_path', str, '/datadisk2/xlxw/Resources/pretrained_models/roberta-base-chinese', 'Bert Pretrained Model Path')
    pretrained_args.add_arg('lm_hidden_size', int, 768, 'HiddenSize of PLM')
    pretrained_args.add_arg('output_hidden_states', bool, True, 'Output PLM Hidden States')
    pretrained_args.add_arg('finetune', bool, True, 'Finetune Or Freeze')

    # Convertor Params
    convertor_args = ArgumentGroup(parser, 'convertor', 'Convertor Settings')
    convertor_args.add_arg('p2next', bool, True, 'Convert Base Point Labels To Next version')
    convertor_args.add_arg('sp_map', bool, False, 'Map special characters (e.g., alphabets, punctuations).')

    # Search Params
    search_params = ArgumentGroup(parser, 'search', 'Search Settings')
    search_params.add_arg('sw_mode', str, 'rsgs', 'Switch Decode Search Mode')
    search_params.add_arg('beam_width', int, 10, 'Beam Width')

    # Model Params
    model_args = ArgumentGroup(parser, 'model', 'Model Settings')
    model_args.add_arg('num_classes', int, 2, 'Number of CGEC classes')
    model_args.add_arg('tagger_classes', int, 6, 'Number of Tagger Classes')
    model_args.add_arg('max_generate', int, 7, 'Number of Max Token Generation')
    model_args.add_arg('padding_size', int, 150, 'Padding Size Of Beet Model')
    model_args.add_arg('padding_val', int, 0, 'Padding Value Of LM Model')
    model_args.add_arg('ignore_val', int, -1, 'Padding Value Of ignore (Generator) index')
    model_args.add_arg('dropout', float, 0.1, 'Dropout')
    model_args.add_arg('scale_attn', bool, True, 'Scale Attention Scores for Pointer Network')
    model_args.add_arg('factorized_embedding', bool, False, 'Factorized Embedding Parameterization')
    model_args.add_arg('lm_emb_size', int, 768, 'LM Model Embedding Size')
    # |- Layer_Attn
    model_args.add_arg('layers_num', int, 12, 'Number Of Bert Layers')
    model_args.add_arg('layer_init_w', float, 0.1, 'Initial Layer Weights')

    # Train Params
    train_args = ArgumentGroup(parser, 'train', 'Training Settings')
    train_args.add_arg('batch_size', int, 64, 'Batch Size')
    train_args.add_arg('shuffle', bool, True, 'DataLoader Shuffle Params')
    train_args.add_arg('droplast', bool, True, 'Drop Rest Data')
    train_args.add_arg('optimizer', str, 'adamW', 'Optimizer Selection, Can Choose [AdamW]')
    train_args.add_arg('lr', float, 1e-5, 'Learning Rate')
    train_args.add_arg('wd', float, 1e-2, 'Weight Decay')
    train_args.add_arg('warmup_steps', int, 10, 'Warm Up Steps Phase')
    train_args.add_arg('epoch', int, 50, 'Epochs')
    train_args.add_arg('criterion', str, 'CE', 'Criterion Selection, Can Choose [CE]')
    train_args.add_arg('print_step', int, 50, 'Training Print Steps')
    train_args.add_arg('eval_step', int, 200, 'Evaluating Steps')

    args = parser.parse_args()
    return args