import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from modules.vgg import Vgg16
from models.o2p2 import O2P2Model

import numpy as np  
import argparse 
from tqdm import trange


class O2P2Trainer(nn.Module):
    def __init__(self, params, model, train_loader, val_loader, logger, results_path, logs_path):
        super(O2P2Trainer, self).__init__()
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.results_path = results_path
        self.logs_path = logs_path

    def train(self):
        for epoch in range(self.params['max_epochs']):
            with trange(len(self.train_loader)) as t:
                self.model.train()
                for idx, sample in enumerate(train_loader):
                    ini_img, fin_img, ini_masks, num_objs = sample['ini_img']


    def eval(self):
        pass

    def compute_perceptual_loss(self):
        pass