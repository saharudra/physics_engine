import torch 
import torch.nn as nn  
from modules.perception import PerceptionModule
from modules.physics import PhysicsTrans, PhysicsInteract
from modules.rendering import RenderModule


class O2P2Model(nn.Module):
    def __init__(self, params):
        super(O2P2Model, self).__init__()
        self.params = params


    def forward(self, ini_img, ini_masks, num_objs):
        """
        TODO: The dataloader is fucking up a bit w.r.t the masks. Check into that.
        """
        pass