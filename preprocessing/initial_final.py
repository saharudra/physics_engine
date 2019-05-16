import numpy as np 
import os 
import argparse
import pickle
import matplotlib.pyplot as plt 

from misc.utils import *

def get_data(params):
    data_root = params['data_root']
    file_path = data_root + 0.p
    init_img = data_root + 0_0.png 
    fi_img = data_root + 0_1.png
    with open(file_path, 'rb') as f:
        configs = pickle.load(f)
    num_masks = len(configs['masks'])
    plt.imshow(plt.imread(init_img))
    plt.show()
    plt.imshow(plt.imread(fi_img))
    plt.show()
    for masks in range(num_masks):
        plt.imshow(configs[masks])
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BroadcastVAE')
    parser.add_argument('--config', type=str, default='/data/Rudra/interaction_modeling/o2p2/physics_engine/configs',
                            help = 'Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)
    print(params)