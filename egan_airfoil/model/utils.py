import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_distance(x, y, diag=False, useCosineDist=False):
    if useCosineDist: # should call similarity function
        return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=2)
    else:
        return torch.sum(torch.abs(x.unsqueeze(0) - y.unsqueeze(1)), 2) # shape [batch, batch]

def strong_convex_func(x, lamb, mean=True, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / lamb / 2.
    else:
        func = torch.exp(x / lamb) / torch.exp(1.) * lamb
    if mean:
        return torch.mean(func)
    else:
        return func

def strong_convex_func_normalized(x, lamb, mean=False, useHingedL2=False): # without lamb
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / 2.
    else:
        func = torch.exp(x / lamb) / torch.exp(1.)

    if mean:
        return torch.mean(func)
    else:
        return func

def sum_probs_func(x, lamb):
    return torch.mean(torch.maximum(x, 0.0)) / lamb

def first_element(input):
    """Improve compatibility of single and multiple output components.
    """
    if type(input) == tuple:
        return input[0]
    else:
        return input