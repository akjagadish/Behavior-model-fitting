import torch

def softmax(z, beta, dim=0):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax inverse temp, small value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    return torch.nn.functional.softmax(torch.squeeze(z * beta), dim=dim)

def pick_action(action_distribution):
        """action selection by sampling from a multinomial.
        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)
        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)
        """
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        return a_t