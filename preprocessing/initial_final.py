import numpy as np 
import os 
import argparse
import pickle
import matplotlib.pyplot as plt 

from misc.utils import *

def get_data(params):
    data_root = params['data_root']
    file_path = data_root + '16.p'
    init_img = data_root + '16_0.png' 
    fi_img = data_root + '16_1.png'
    with open(file_path, 'rb') as f:
        configs = pickle.load(f)
    print(configs['masks'])
    import pdb; pdb.set_trace()
    num_masks = len(configs['masks'])
    plt.imsave('init_img.jpg', plt.imread(init_img))
    plt.imsave('fi_img.jpg', plt.imread(fi_img))
    for mask in range(num_masks):
        print(list(configs['masks'].values())[mask].shape)
        plt.imsave('mask_' + str(mask) + '.jpg', list(configs['masks'].values())[mask])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
    parser.add_argument('--config', type=str, default='/data/Rudra/interaction_modeling/o2p2/physics_engine/configs/pre-planning.yml',
                            help = 'Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)
    print(params)
    get_data(params)
