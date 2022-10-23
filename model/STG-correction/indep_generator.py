import os
import warnings
warnings.filterwarnings("ignore")
from config import generator_parse
# Import torch
from torch.utils.data import DataLoader
import torch
# Import Dataset
from DataProcessor import GeneratorDataset
# Import utils
from utils import get_device, set_seed, collate_fn_base as collate_fn, export_generator
# Import Trainer
from Trainer import GeneratorTrainer
# Import Model
from Model import GeneratorModel
from transformers import AdamW

def train(args, check_dirname = ""):
    print('=' * 20 + 'Train Generator Module' + '=' * 20)
    # Checkpoint
    check_dir = args.checkpoints if check_dirname == "" else os.path.join(args.checkpoints, check_dirname)
    if os.path.exists(check_dir) is not True:
        os.mkdir(check_dir)
        print('>> Create Checkpoint Dir at %s' % check_dir)
    # Dataset
    if os.path.exists(os.path.join(args.data_base_dir, 'train_generator.pt')) is not True:
        train_dir = os.path.join(args.data_base_dir, 'train.csv')
        Trainset = GeneratorDataset(args, train_dir, 'train')
        torch.save(Trainset, os.path.join(args.data_base_dir, 'train_generator.pt'))
    else:
        Trainset = torch.load(os.path.join(args.data_base_dir, 'train_generator.pt'))
        print('Direct Load Train Dataset')
    if os.path.exists(os.path.join(args.data_base_dir, 'valid_generator.pt')) is not True:
        valid_dir = os.path.join(args.data_base_dir, 'valid.csv')
        Validset = GeneratorDataset(args, valid_dir, 'valid')
        torch.save(Validset, os.path.join(args.data_base_dir, 'valid_generator.pt'))
    else:
        Validset = torch.load(os.path.join(args.data_base_dir, 'valid_generator.pt'))
        print('Direct Load Valid Dataset')
    # DataLoader
    TrainLoader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.droplast, collate_fn=collate_fn)
    ValidLoader = DataLoader(Validset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.droplast, collate_fn=collate_fn)
    # Device
    device = get_device(args.cuda, args.gpu_id)
    # Model
    model = GeneratorModel(args, device).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    criterion = torch.nn.NLLLoss().to(device)
    trainer   = GeneratorTrainer(args, model, criterion, optimizer, device, check_dir)
    trainer.train(TrainLoader, ValidLoader)
    print('>>> Experiments for generator module has done.')

if __name__ == '__main__':
    args = generator_parse()
    checkp = args.checkp
    args.lm_hidden_size = 768
    set_seed(args)

    if args.mode == 'train':
        train(args, checkp)
    else:
        raise Exception('Error mode :`{}`.'.format(args.mode))