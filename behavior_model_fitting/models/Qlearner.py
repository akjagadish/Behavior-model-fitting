import torch 
from torch import nn
from behavior_model_fitting.helpers.policy import softmax
import torch.nn.functional as F

# TODO: 
# vectorise this to take batch of participants by taking a batch of lr, betas 
# bayesQlearner and Qlearner can be merged
# policy within the q-learner or seperate? (design choice)

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
        assert self.beta > 0. 
        return F.log_softmax(x*self.beta, dim=dim) #torch.log(softmax(x, self.beta, dim))

class bayesQlearner(nn.Module):

    def __init__(self, parameters, Qinit=[0.5, 0.5]):
       super().__init__()  
       self.lr = nn.Parameter(torch.tensor(parameters.get("lr",torch.rand(1)*0.100)))
       self.beta = nn.Parameter(torch.tensor(parameters.get("beta",torch.rand(1)*10))) 
       self.Qinit = torch.tensor(Qinit)
       
    def forward(self, a_t, r_t, Q=None):
       Q  = self.Qinit if Q is None else Q
       Q = Q + self.lr*(r_t - Q)
       return Q

    def log_softmax(self, x, dim=0):
        return F.log_softmax(x*self.beta, dim=dim)  #torch.log(softmax(x, self.beta, dim))



