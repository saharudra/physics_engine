import torch 
import torch.nn as nn  

class PhysicsTrans(nn.Module):
    """
    Defining unary potentials for transistion of an object vector
    obj_vec: [batch_size, obj_dim]
    """
    def __init__(self, params):
        super(PhysicsTrans, self).__init__()
        self.params = params
        self.unary_params = self.params['physics-transition']

        self.unary_potential_network()

    def unary_potential_network():
        self.unary_transition = nn.Sequential(
            nn.Linear(self.unary_params['obj_dim'], self.unary_params['hidden']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.unary_params['hidden'], self.unary_params['hidden']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.unary_params['hidden'], self.unary_params['obj_dim'])
        )

    def forward(self, obj_vec):
        trans_obj_vec = self.unary_transition(obj_vec)
        return trans_obj_vec


class PhysicsInteract(nn.Module):
    """
    Defining binary potentials between object interactions.
    obj_vec: [batch_size, obj_dim]
    """
    def __init__(self, params):
        super(PhysicsInteract, self).__init__()
        self.params = params
        self. interaction_params = self.params['physics-transition']

        self.pairwise_potential_network()

    def pairwise_potential_network():
        self.interact_transition = nn.Sequential(
            nn.Linear(self.interaction_params['hidden'], self.interaction_params['hidden']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.interaction_params['hidden'], self.interaction_params['hidden']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.interaction_params['hidden'], self.interaction_params['obj_dim'])
        )

    def forward(self, obj_vec_1, obj_vec_2):
        # Concatenate the two object vectors
        obj_vec = torch.cat((obj_vec_1, obj_vec_2), 1)
        interact_obj_vec = self.interact_transition(obj_vec)
        return interact_obj_vec
