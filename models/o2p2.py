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
        TODO: The forward pass through the model will give the outputs required for loss calculation.
              Trainer will work on this model, thus changing the model, when replacing this with a batched run
              will be easier this way.
        """
        pass