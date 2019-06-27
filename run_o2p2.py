import torch
import torch.nn as nn   

from models.o2p2 import O2P2Model
from trainers.o2p2_trainer import O2P2Trainer
from dataloaders.initial_final import initial_final_dataloader
from misc.utils import *
from misc.logger import Logger

import pprint
import argparse
import numpy as np 
import time
import datetime

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')

parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
parser.add_argument('--config', type=str, default='/home/rudra/Downloads/rudra/relationship_modeling/o2p2/physics_engine/configs/pre-planning.yml',
                        help = 'Path to config file')
opts = parser.parse_args()
params = get_config(opts.config)
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(params)

# Define models and dataloaders
train_loader, val_loader = initial_final_dataloader(params)
model = O2P2Model(params)

if params['use_cuda']:
    model = model.cuda()

exp_results_path = params['project_root'] + '/results/' + params['exp_name'] + '_' + timestamp + '/'
exp_logs_path = params['project_root'] + '/logs/' + params['exp_name'] + '_' + timestamp + '/'
mkdir_p(exp_logs_path)
mkdir_p(exp_results_path)

logger = Logger(exp_logs_path)

trainer = O2P2Trainer(params, model, train_loader, val_loader, logger, exp_results_path, exp_logs_path)

trainer.train()