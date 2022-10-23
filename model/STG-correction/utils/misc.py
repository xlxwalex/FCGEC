import torch
import numpy as np
import random

# Device
def _get_device(cuda : bool, gpu_id : int = 0) -> torch.device:
    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available() and gpu_id < gpu_count:
        device = torch.device("cuda:" + str(gpu_id) if cuda else "cpu")
    else:
        device = torch.device("cpu")
    return device

# Seed
def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)