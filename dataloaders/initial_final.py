import torch 
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import os 
import numpy as np 
import matplotlib.pyplot as plt 
from misc.utils import *
import argparse
import pickle

class InitialFinal(Dataset):
    def __init__(self, params, partition='train', transform=None):
        super(InitialFinal, self).__init__()
        self.params = params
        self.partition = partition
        self.root = self.params['data_root'] + '/' + self.partition + '/'
        self.transform = transform

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])  # ITU-R 601-2 luma transform

    def __getitem__(self, idx):
        # Get initial image, convert to tensor, permute dims and obtain the rgb sample
        ini_img_mask_filepath = self.root + str(idx) + '_0.png'
        ini_img_mask = torch.from_numpy(plt.imread(ini_img_mask_filepath)).permute(2, 0, 1)  
        rgb_ini_img = ini_img_mask[:3, :, :]

        # Get final image, convert to tensor, permute dims and obtain the rgb sample
        fin_img_mask_filepath = self.root + str(idx) + '_1.png'
        fin_img_mask = torch.from_numpy(plt.imread(fin_img_mask_filepath)).permute(2, 0, 1)
        rgb_fin_img = fin_img_mask[:3, :, :]

        # Read config file to obtain initial and final masks
        with open(self.root + str(idx) + '.p', 'rb') as pf:
            config = pickle.load(pf)
        curr_masks = list(config['masks'].values())
        num_objects = len(curr_masks)
        left_objects = self.params['max_objects'] - num_objects
        ini_masks = torch.Tensor().float()
        fin_masks = torch.Tensor().float()

        # Add initial and final masks to the tensor.
        for mask in curr_masks:
            curr_ini_mask = torch.from_numpy(self.rgb2gray(mask[0])).unsqueeze(0).float()
            curr_fin_mask = torch.from_numpy(self.rgb2gray(mask[1])).unsqueeze(0).float()
            ini_masks = torch.cat((ini_masks, curr_ini_mask), 0)
            fin_masks = torch.cat((fin_masks, curr_fin_mask), 0)

        # Add all 0 masks to the tensor
        zero_mask = torch.zeros(1, self.params['img_h'], self.params['img_w']).float()
        for _ in range(left_objects):
            ini_masks = torch.cat((ini_masks, zero_mask), 0)
            fin_masks = torch.cat((fin_masks, zero_mask), 0)

        # Define sample dictionary and add the required inputs.
        sample = {}
        sample['ini_img'] = rgb_ini_img
        sample['fin_img'] = rgb_fin_img
        sample['ini_masks'] = ini_masks
        sample['fin_masks'] = fin_masks
        sample['num_objs'] = num_objects

        return sample

    def __len__(self):
        if self.partition == 'train':
            return (len(os.listdir(self.params['data_root'] + '/train/')) - 1) // self.params['num_files']  # There is an extra metadata file in the folder.
        elif self.partition == 'val':
            return (len(os.listdir(self.params['data_root'] + '/val/')) - 1) // self.params['num_files']  # There is an extra metadata file in the folder


def initial_final_dataloader(params):
    """
    TODO: Currently transforming the numpy arrays to tensor inside the __getitem__ function.
    Use a ToTensor() class when batching this.
    """
    # trans = [ToTensor()]
    # train_set = InitialFinal(params, partition='train', transform=transforms.Compose(trans))
    # val_set = InitialFinal(params, partition='val', transform=transforms.Compose(trans))

    train_set = InitialFinal(params, partition='train', transform=None)
    val_set = InitialFinal(params, partition='val', transform = None)

    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['use_cuda']}

    train_loader = DataLoader(dataset=train_set,
                              batch_size=params['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=val_set,
                             batch_size=params['batch_size'], shuffle=False, **kwargs)
    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
    parser.add_argument('--config', type=str, default='/home/rudra/Downloads/rudra/relationship_modeling/o2p2/physics_engine/configs/pre-planning.yml',
                            help = 'Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    train_loader, val_loader = initial_final_dataloader(params)

    for idx, sample in enumerate(train_loader):
        ini_img = sample['ini_img']
        fin_img = sample['fin_img']
        import pdb; pdb.set_trace()
