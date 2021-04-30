import torch
import numpy as np

def nll_per_model_per_subject(outcomes, q, lr, beta, choice, ntrials=100, return_qvalues=False):
    nll = 0.
    qvalues = []
    for trial in range(ntrials):
        a_t = choice[trial]
        nll += -(q[a_t]*beta - torch.log(torch.exp(q*beta).sum()))
        r_t = outcomes[trial][a_t]
        q[a_t] = q[a_t] + lr*(r_t-q[a_t])
        qvalues.append(q.clone().detach())
    if return_qvalues:
        return nll
    else:
        return nll, qvalues