import torch 
import torch.nn as nn  

import argparse
from misc.utils import * 

class RenderModule(nn.Module):
    def __init__(self, params):
        super(RenderModule, self).__init__()
        self.params = params
        self.render_params = self.params['render']

        self.render_module()

    def render_module(self):
        self.render_rgb_network = nn.Sequential(
            nn.ConvTranspose2d(self.render_params['obj_dim'], self.render_params['obj_dim'] // 2, self.render_params['ks1'], self.render_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.render_params['obj_dim'] // 2),
            nn.ConvTranspose2d(self.render_params['obj_dim'] // 2, self.render_params['obj_dim'] // 4, self.render_params['ks1'], self.render_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.render_params['obj_dim'] // 4),
            nn.ConvTranspose2d(self.render_params['obj_dim'] // 4, self.render_params['obj_dim'] // 8, self.render_params['ks2'], self.render_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.render_params['obj_dim'] // 8),
            nn.ConvTranspose2d(self.render_params['obj_dim'] // 8, self.render_params['oc'], self.render_params['ks2'], self.render_params['stride'])
        )

        self.render_mask_network = nn.Sequential(
            nn.ConvTranspose2d(self.render_params['obj_dim'], self.render_params['obj_dim'] // 2, self.render_params['ks1'], self.render_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.render_params['obj_dim'] // 2),
            nn.ConvTranspose2d(self.render_params['obj_dim'] // 2, self.render_params['obj_dim'] // 4, self.render_params['ks1'], self.render_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.render_params['obj_dim'] // 4),
            nn.ConvTranspose2d(self.render_params['obj_dim'] // 4, self.render_params['obj_dim'] // 8, self.render_params['ks2'], self.render_params['stride']),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.render_params['obj_dim'] // 8),
            nn.ConvTranspose2d(self.render_params['obj_dim'] // 8, self.render_params['mc'], self.render_params['ks2'], self.render_params['stride']),
            nn.Sigmoid()
        )

    def forward(self, obj_vec):
        obj_feature = obj_vec.view(-1, self.render_params['obj_dim'], 1, 1)
        obj_rgb_img = self.render_rgb_network(obj_feature)
        obj_mask_img = self.render_mask_network(obj_feature)
        return obj_rgb_img, obj_mask_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
    parser.add_argument('--config', type=str, default='/home/rudra/Downloads/rudra/relationship_modeling/o2p2/physics_engine/configs/pre-planning.yml',
                            help = 'Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config) 

    render_rgb_module = RenderModule(params)
    obj_vec = torch.randn(1, 256)

    obj_rgb_img, obj_mask_img = render_rgb_module(obj_vec)
    print(obj_rgb_img.size(), obj_mask_img.size())
