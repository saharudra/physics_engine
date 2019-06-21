import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np  
import argparse 


class O2P2Trainer(nn.Module):
    def __init__(self, params, train_loader, val_loader, logger, results_path, logs_path):
        super(O2P2Trainer, self).__init__()
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.results_path = results_path
        self.logs_path = logs_path

    def train(self):
        pass

    def eval(self):
        pass

    def compute_perceptual_loss(self):
        pass