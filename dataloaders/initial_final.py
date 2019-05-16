import torch 
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os 
import numpy as np 
import matplotlib.pyplot as plt 
from misc.utils import *
import argparse

class InitialFinal(Dataset):
    def __init__(self, params, partition='train', transform=None):
        super(InitialFinal, self).__init__()
        self.params = params
        self.root = self.params['data_root']
        self.transform = transform    
        self.partition = partition

    def __getitem__(self, idx):
        initial_image = plt.imread(self.root + str(idx) + '_0.png')
        final_image = plt.imread(self.root + str(idx) + '_1.png')
        with open(self.root + str(idx) + '.p', 'rb') as pf:
            config
    
    def __len__(self):
        if self.partition == 'train':
            return self.params['num_train_images']
        elif self.partition = 'val':
            return self.params['num_val_images']
    