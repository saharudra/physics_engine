import torch 
import torch.nn as nn  

class PhysicsTrans(nn.Module):
    """
    Defining unary potentials for transistion of an object vector
    """
    def __init__(self, params):
        super(PhysicsTrans, self).__init__()
        self.params = params


class PhysicsInteract(nn.Module):
    """
    Defining binary potentials between object interactions.
    """
    def __init__(self, params):
        super(PhysicsInteract, self).__init__()
        self.params = params