from Trainer.SwitchTrainer import SwitchTrainer, SwitchTrainerTTI
from Trainer.TaggerTrainer import TaggerTrainer, TaggerTrainerTTI
from Trainer.GeneratorTrainer import GeneratorTrainer
from Trainer.JointTrainer import JointTrainer
from Trainer.Trainer import Trainer

__all__ = ['Trainer', 'SwitchTrainer', 'SwitchTrainerTTI', # Switch Module
          'TaggerTrainer', 'TaggerTrainerTTI',  # Tagger Module
           'GeneratorTrainer', "JointTrainer" # Generator Modeule
          ]