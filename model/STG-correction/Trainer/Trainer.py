# Import Libs
from torch.utils.data import DataLoader

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device, checkp, scheduler = None):
        super(Trainer, self).__init__()
        self.args       = args
        # Training Component
        self.model       = model
        self.criterion   = criterion
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        # Training Params
        self.checkpoint  = checkp
        self.epoch       = 0
        self.step        = 0
        self.best        = - float('inf')
        self.eval_inform = []

    # Train Model
    def train(self, Trainset : DataLoader, Validset : DataLoader):
        raise NotImplementedError

    # Valid Model
    def valid(self, Validset : DataLoader):
        raise NotImplementedError

    # Test Model
    def test(self, Testset: DataLoader):
        raise NotImplementedError

    # Generate Checkpoints
    def _generate_checkp(self) -> dict:
        checkpoints = {
            'model': self.model.state_dict(),
            'optim': self.optimizer,
            'metric': self.eval_inform,
            'args': self.args,
            'epoch': self.epoch
        }
        return checkpoints
