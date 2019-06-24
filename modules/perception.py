import torch 
import torch.nn as nn 

import argparse
from misc.utils import *

class PerceptionModule(nn.Module):
    def __init__(self, params):
        super(PerceptionModule, self).__init__()
        self.params = params
        self.perception_params = self.params['perception']

        self.perception_module()

    def perception_module(self):
        self.perception_conv_network = nn.Sequential(
            nn.Conv2d(self.perception_params['ic'], self.perception_params['npf'], self.perception_params['ks'], self.perception_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.perception_params['npf']),
            nn.Conv2d(self.perception_params['npf'], self.perception_params['npf'] * 2, self.perception_params['ks'], self.perception_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.perception_params['npf'] * 2),
            nn.Conv2d(self.perception_params['npf'] * 2, self.perception_params['npf'] * 4, self.perception_params['ks'], self.perception_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.perception_params['npf'] * 4),
            nn.Conv2d(self.perception_params['npf'] * 4, self.perception_params['npf'] * 8, self.perception_params['ks'], self.perception_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.perception_params['npf'] * 8)
        )

        self.perception_fc_network = nn.Sequential(
            nn.Linear(self.perception_params['npf'] * 8 * 2 * 2, self.perception_params['obj_dim'])
        )

    def forward(self, img, mask):
        # Depth-wise concatenate the image and the mask
        # image: [1, 3, 64, 64], mask: [1, 1, 64, 64]
        perception_input = torch.cat((img, mask), 1)
        object_feature = self.perception_conv_network(perception_input)
        feature_size = object_feature.size()
        object_feature = object_feature.view(-1, feature_size[1] * feature_size[2] * feature_size[3])
        object_vector = self.perception_fc_network(object_feature)
        return object_vector


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
#     parser.add_argument('--config', type=str, default='/home/rudra/Downloads/rudra/relationship_modeling/o2p2/physics_engine/configs/pre-planning.yml',
#                             help = 'Path to config file')
#     opts = parser.parse_args()
#     params = get_config(opts.config) 

#     perception_module = PerceptionModule(params)
#     img = torch.randn(1, 3, 64, 64)
#     mask = torch.randn(1, 1, 64, 64)

#     obj_vec = perception_module(img, mask)
#     print(obj_vec.size())