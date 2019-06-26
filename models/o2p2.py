import torch 
import torch.nn as nn  
from modules.perception import PerceptionModule
from modules.physics import PhysicsTrans, PhysicsInteract
from modules.rendering import RenderModule

import argparse
from torchvision.utils import save_image
from misc.utils import *
from dataloaders.initial_final import initial_final_dataloader


class O2P2Model(nn.Module):
    def __init__(self, params):
        super(O2P2Model, self).__init__()
        self.params = params

        self.perception_moddule = PerceptionModule(self.params)
        self.render_module = RenderModule(self.params)
        self.physics_transistion_module = PhysicsTrans(self.params)
        self.physics_interaction_module = PhysicsInteract(self.params)


    def forward(self, ini_img, ini_masks, num_objs):
        """
        TODO: The forward pass through the model will give the outputs required for loss calculation.
              Trainer will work on this model, thus changing the model, when replacing this with a batched run
              will be easier this way.

              Currently dealing with a batch size of 1.
        """
        # Get initial object vectors: [num_objs, obj_dim]
        ini_obj_vec = torch.Tensor().float()
        if self.params['use_cuda']:
            ini_obj_vec = ini_obj_vec.cuda()
        for obj in range(num_objs.item()):
            curr_obj_mask = ini_masks[:, obj:obj+1, :, :]
            curr_obj_vec = self.perception_moddule(ini_img, curr_obj_mask)
            ini_obj_vec = torch.cat((ini_obj_vec, curr_obj_vec), 0)
        
        # Reconstruct initial image
        recon_ini_img = torch.zeros((ini_img.size())).float()
        if self.params['use_cuda']:
            recon_ini_img = recon_ini_img.cuda()
        for obj in range(num_objs.item()):
            curr_obj_rgb, curr_obj_mask = self.render_module(ini_obj_vec[obj:obj+1, :])
            curr_rgb_img = curr_obj_rgb * curr_obj_mask
            recon_ini_img = recon_ini_img + curr_rgb_img
            
        # Get transistions of object vectors
        trans_obj_vec = torch.Tensor().float()
        if self.params['use_cuda']:
            trans_obj_vec = trans_obj_vec.cuda()
        for obj in range(num_objs.item()):
            curr_trans_obj_vec = self.physics_transistion_module(ini_obj_vec[obj:obj+1, :])
            trans_obj_vec = torch.cat((trans_obj_vec, curr_trans_obj_vec), 0)

        # Get interactions of object vectors
        interact_obj_vec = torch.Tensor().float()
        if self.params['use_cuda']:
            interact_obj_vec = interact_obj_vec.cuda()
        for curr_obj in range(num_objs.item()):
            curr_interact_obj_vec = torch.zeros((1, self.params['perception']['obj_dim']))
            for other_obj in range(num_objs.item()):
                if not(curr_obj == other_obj):
                    temp_obj_vec = self.physics_interaction_module(ini_obj_vec[curr_obj:curr_obj+1, :], ini_obj_vec[other_obj:other_obj+1, :])
                    curr_interact_obj_vec += temp_obj_vec
            interact_obj_vec = torch.cat((interact_obj_vec, curr_interact_obj_vec), 0)


        # Get final object vector 
        # o_f = o_trans + o_interact + o_ini
        fin_obj_vec = torch.Tensor().float()
        if self.params['use_cuda']:
            fin_obj_vec = fin_obj_vec.cuda()
        for obj in range(num_objs.item()):
            curr_fin_obj_vec = ini_obj_vec[obj:obj+1, :] + interact_obj_vec[obj:obj+1, :] + trans_obj_vec[obj:obj+1, :]
            fin_obj_vec = torch.cat((fin_obj_vec, curr_fin_obj_vec), 0)

        # Reconstruct transition image
        recon_fin_img = torch.zeros((ini_img.size())).float()
        if self.params['use_cuda']:
            recon_fin_img = recon_fin_img.cuda()
        for obj in range(num_objs.item()):
            fin_obj_rgb, fin_obj_mask = self.render_module(fin_obj_vec[obj:obj+1, :])
            fin_rgb_img = fin_obj_rgb * fin_obj_mask
            recon_fin_img = recon_fin_img + fin_rgb_img

        return recon_ini_img, recon_fin_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='O2P2 initial final preprocessing')
    parser.add_argument('--config', type=str, default='/home/rudra/Downloads/rudra/relationship_modeling/o2p2/physics_engine/configs/pre-planning.yml',
                            help = 'Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config) 

    model = O2P2Model(params)

    train_loader, val_loader = initial_final_dataloader(params)

    for idx, sample in enumerate(train_loader):
        ini_img = sample['ini_img']
        ini_masks = sample['ini_masks']
        num_objs = sample['num_objs']
        model(ini_img, ini_masks, num_objs)
        # save_image(ini_img, 'ini_img.png')
        # save_image(ini_masks[:, 0:1, :, :], 'curr_obj_mask.png')
        # import pdb; pdb.set_trace()

