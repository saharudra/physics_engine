import torch 
import torch.nn as nn  
from modules.perception import PerceptionModule
from modules.physics import PhysicsTrans, PhysicsInteract
from modules.rendering import RenderModule


class O2P2Model(nn.Module):
    def __init__(self, params):
        super(O2P2Model, self).__init__()
        self.params = params