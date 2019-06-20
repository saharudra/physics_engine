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
        """
        Currently batching everything to create tensors with number of masks equal to
        the max-objects that we can expect out of this.
        """
        super(InitialFinal, self).__init__()
        self.params = params
        self.partition = partition
        self.root = self.params['data_root'] + '/' + self.partition + '/'
        self.transform = transform    

    def __getitem__(self, idx):
        # Get initial and final images
        initial_image = plt.imread(self.root + str(idx) + '_0.png')
        print("Initial image shape: {}".format(initial_image.shape))
        # initial_image = initial_image[:, :, :3]
        final_image = plt.imread(self.root + str(idx) + '_1.png')
        with open(self.root + str(idx) + '.p', 'rb') as pf:
            config =  pickle.load(pf)
            curr_masks = config['masks'].values()
        curr_objects_meta = torch.zeros((self.params['max_objects']))
        num_objects = len(curr_masks)
        # Get object meta-mask and object meta-num-objects
        curr_objects_meta[:num_objects] = 1.0
        curr_num_objects = torch.tensor((num_objects))
        empty_mask = torch.zeros((self.params['mask_ic'],self.params['mask_h'], self.params['mask_w']))

        # Get initial and final masks
        initial_masks = torch.Tensor()
        final_masks = torch.Tensor()

        # Depth-wise concatenate masks for objects that are present
        # print("Number of objects are {}".format(num_objects))
        for obj in range(num_objects):
            curr_obj_inital_mask = torch.from_numpy(list(curr_masks)[obj][0]).unsqueeze(0)
            # print("curr object initial mask shape")
            # print(curr_obj_inital_mask.shape)
            curr_obj_final_mask = torch.from_numpy(list(curr_masks)[obj][1]).unsqueeze(0)
            initial_masks = torch.cat((initial_masks.float(), curr_obj_inital_mask.float()))
            final_masks = torch.cat((final_masks.float(), curr_obj_final_mask.float()))

        # # Depth-wise concatenate empty masks for objects not present
        # for non_obj in range(self.params['max_objects'] - num_objects):
        #     initial_masks = torch.cat((initial_masks, empty_mask))
        #     final_masks = torch.cat((final_masks, empty_mask))

        sample = {}
        sample['ini_img'] = initial_image
        sample['fin_img'] = final_image
        sample['ini_masks'] = initial_masks
        sample['fin_masks'] = final_masks
        sample['num_objects'] = curr_num_objects
        sample['object_mask_meta'] = curr_objects_meta

        if self.transform:
            sample = self.transform(sample)

        return sample           
    
    def __len__(self):
        if self.partition == 'train':
            return (len(os.listdir(self.params['data_root'] + '/train/')) - 1) // self.params['num_files']  # There are three files for each of the instance
        elif self.partition == 'val':
            return (len(os.listdir(self.params['data_root'] + '/val/')) - 1) // self.params['num_files']


class ToTensor(object):
    """
    Converts the intial and final images to tensors and divides them by 255.
    Mostly a placeholder as the functionality can be transferred inside the __getitem__
    method above.
    """
    def __call__(self, sample):
        initial_image = sample['ini_img']
        final_image = sample['fin_img']
        initial_image = torch.from_numpy(initial_image).float().div(255).permute(2, 0, 1)
        final_image = torch.from_numpy(final_image).float().div(255).permute(2, 0, 1)
        sample['ini_img'] = initial_image
        sample['fin_img'] = final_image
        return sample


def initial_final_dataloader(params):
    trans = [ToTensor()]

    train_set = InitialFinal(params, partition='train', transform=transforms.Compose(trans))
    # val_set = InitialFinal(params, partition='val', transform=transforms.Compose(trans))

    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['use_cuda']}

    train_loader = DataLoader(dataset=train_set,
                              batch_size=params['batch_size'], shuffle=True, **kwargs)
    # val_loader = DataLoader(dataset=val_set,
    #                          batch_size=params['batch_size'], shuffle=False, **kwargs)
    return train_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
    parser.add_argument('--config', type=str, default='/home/rudra/Downloads/rudra/relationship_modeling/o2p2/physics_engine/configs/pre-planning.yml',
                            help = 'Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    train_loader = initial_final_dataloader(params)

    for idx, sample in enumerate(train_loader):
        ini_img, fin_img, ini_masks, fin_masks, num_obects, object_mask_meta = sample['ini_img'], sample['fin_img'], sample['ini_masks'], sample['fin_masks'], sample['num_objects'], sample['object_mask_meta']
        print("The weird line {}".format(ini_img.shape))
        save_image(ini_img, 'ini_img.png')
        save_image(fin_img, 'fin_img.png')
        for obj in range(num_objects):
            save_image(ini_masks[obj], 'ini_masks_' + str(obj) + '.png')
            save_image(fin_masks[obj], 'fin_masks_' + str(obj) + '.png')
        print("This is sanity check for data loading")
        import pdb; pdb.set_trace()