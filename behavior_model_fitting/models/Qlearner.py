import torch 
from torch import nn
#from behavior_model_fitting.helpers.policy import softmax
import torch.nn.functional as F

# TODO: 
# vectorise this to take batch of participants by taking a batch of lr, betas 
# bayesQlearner and Qlearner can be merged
# policy within the q-learner or seperate? (design choice)
 #torch.log(softmax(x, self.beta, dim))
class Qlearner(nn.Module):

    def __init__(self, *args, Qinit=[0.5, 0.5], lr=0.01, beta=10., **kwargs):
       super().__init__()  
       self.lr = nn.Parameter(torch.tensor(lr)) 
       self.beta = nn.Parameter(torch.tensor(beta))
       self.Qinit = torch.tensor(Qinit).detach().clone()
       
    def forward(self, r_t, Q=None):
        Q  = self.Qinit if Q is None else Q
        Q = Q + self.lr*(r_t - Q)
        return Q

    def log_softmax(self, x, dim=0):
        return F.log_softmax(x*self.beta, dim=dim)  
   
s
