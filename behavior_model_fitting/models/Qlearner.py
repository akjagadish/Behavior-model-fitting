import torch 
from torch import nn
from behavior_model_fitting.helpers.policy import softmax

class Qlearner(nn.Module):

    def __init__(self, Qinit=[0.5, 0.5]):
       super().__init__()
       self.lr = nn.Parameter(torch.rand(1)*0.100) 
       self.beta = nn.Parameter(torch.rand(1)*10) 
       self.Qinit = torch.tensor(Qinit)
       
    def forward(self, a_t, r_t, Q=None):
       Q  = self.Qinit if Q is None else Q
       Q = Q + self.lr*(r_t - Q)
       return Q

    def log_softmax(self, x, dim=0):
        return torch.log(softmax(x, self.beta, dim))




