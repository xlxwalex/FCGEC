import os
import warnings
warnings.filterwarnings("ignore")
from config import switch_parse
# Import torch
from torch.utils.data import DataLoader
import torch
# Import Dataset
from DataProcessor import SwitchDatasetWTTI as SwitchDataset
# Import utils
from utils.collate import collate_fn_bertbase_tti as collate_fn
from utils import get_device, set_seed
# Import Trainer
from Trainer import SwitchTrainerTTI as SwitchTrainer
# Import Model
from Model import SwitchModelTTI as SwitchModel
from transformers import AdamW

def train(args, check_dirname = ""):
    print('=' * 20 + 'Train Switch+TTI Module' + '=' * 20)
    # Checkpoint
    check_dir = args.checkpoints if check_dirname == "" else os.path.join(args.checkpoints, check_dirname)
    if os.path.exists(check_dir) is not True:
        os.mkdir(check_dir)
        print('>> Create Checkpoint Dir at %s' % check_dir)
    # Dataset
    if os.path.exists(os.path.join(args.data_base_dir, 'train_switch.pt')) is not True:
        train_dir = os.path.join(args.data_base_dir, 'train.csv')
        Trainset = SwitchDataset(args, train_dir, 'train')
        torch.save(Trainset, os.path.join(args.data_base_dir, 'train_switch.pt'))
    else:
        Trainset = torch.load(os.path.join(args.data_base_dir, 'train_switch.pt'))
        print('Direct Load Train Dataset')
    if os.path.exists(os.path.join(args.data_base_dir, 'valid_switch.pt')) is not True:
        valid_dir = os.path.join(args.data_base_dir, 'valid.csv')
        Validset = SwitchDataset(args, valid_dir, 'valid')
        torch.save(Validset, os.path.join(args.data_base_dir, 'valid_switch.pt'))
    else:
        Validset = torch.load(os.path.join(args.data_base_dir, 'valid_switch.pt'))
        print('Direct Load Valid Dataset')
    # DataLoader
    TrainLoader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.droplast, collate_fn=collate_fn)
    ValidLoader  = DataLoader(Validset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=args.droplast, collate_fn=collate_fn)
    # Device
    device = get_device(args.cuda, args.gpu_id)
    # Model
    model = SwitchModel(args, device).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    criterion_pointer = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_val).to(device)
    criterion_tti = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.3, 0.7])).to(device)
    criterion = (criterion_pointer, criterion_tti)
    trainer   = SwitchTrainer(args, model, criterion, optimizer, device, check_dir)
    trainer.train(TrainLoader, ValidLoader)
    print('>>> Experiments for switch module has done.')

if __name__ == '__main__':
    args = switch_parse()
    checkp = args.checkp
    set_seed(args)
    if args.mode == 'train':
        train(args, checkp)
    else:
        raise Exception('Error mode :`{}`.'.format(args.mode))