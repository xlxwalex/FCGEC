import os
import warnings
warnings.filterwarnings("ignore")
from config import joint_parse
# Import torch
from torch.utils.data import DataLoader
import torch
# Import Dataset
from DataProcessor import JointDataset
# Import utils
from utils.collate import collate_fn_jointV2 as collate_fn
from utils import get_device, set_seed, TAGGER_MAP
# Import Trainer
from Trainer import JointTrainer
# Import Model
from Model import JointModelwithEncoder
from transformers import AdamW

def train(args, check_dirname = ""):
    print('=' * 20 + 'Train Joint Model' + '=' * 20)
    # Process Args
    args.tagger_classes = len(TAGGER_MAP.keys())
    # Checkpoint
    check_dir = args.checkpoints if check_dirname == "" else os.path.join(args.checkpoints, check_dirname)
    # Dataset
    if os.path.exists(check_dir) is not True:
        os.mkdir(check_dir)
        print('>> Create Checkpoint Dir at %s' % check_dir)
    # Dataset
    if os.path.exists(os.path.join(args.data_base_dir, 'train.pt')) is not True:
        train_dir = os.path.join(args.data_base_dir, 'train.csv')
        Trainset = JointDataset(args, train_dir, 'train')
        torch.save(Trainset, os.path.join(args.data_base_dir, 'train.pt'))
    else:
        Trainset = torch.load(os.path.join(args.data_base_dir, 'train.pt'))
        print('Direct Load Train Dataset')
    if os.path.exists(os.path.join(args.data_base_dir, 'valid.pt')) is not True:
        valid_dir = os.path.join(args.data_base_dir, 'valid.csv')
        Validset = JointDataset(args, valid_dir, 'valid')
        torch.save(Validset, os.path.join(args.data_base_dir, 'valid.pt'))
    else:
        Validset = torch.load(os.path.join(args.data_base_dir, 'valid.pt'))
        print('Direct Load Valid Dataset')
    # DataLoader
    TrainLoader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.droplast,collate_fn=collate_fn)
    ValidLoader = DataLoader(Validset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.droplast,collate_fn=collate_fn)
    # Device
    device = get_device(args.cuda, args.gpu_id)
    # Model
    model = JointModelwithEncoder(args, device).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    criterion = {
        'sw'  : torch.nn.CrossEntropyLoss(ignore_index=args.ignore_val).to(device),
        'gen' : torch.nn.NLLLoss().to(device),
        'tag' : (torch.nn.CrossEntropyLoss(reduction = 'sum').to(device),
                 torch.nn.CrossEntropyLoss(reduction = 'sum', ignore_index=args.ignore_val).to(device))

    }
    trainer = JointTrainer(args, model, criterion, optimizer, device, check_dir)
    trainer.train(TrainLoader, ValidLoader)
    print('>>> Experiments for joint model has done.')

if __name__ == '__main__':
    args = joint_parse()
    args.tagger_classes = len(TAGGER_MAP.keys())
    checkp = args.checkp
    set_seed(args)
    if args.mode == 'train':
        train(args, checkp)
    else:
        raise Exception('Error mode :`{}`.'.format(args.mode))