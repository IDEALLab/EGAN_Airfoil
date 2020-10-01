import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_distance(x, y, diag=False, useCosineDist=False):
    x = x.reshape([len(x), -1]); y = y.reshape([len(y), -1])
    if useCosineDist: # should call similarity function
        return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=-1)
    else:
        return torch.sum(torch.abs(x.unsqueeze(0) - y.unsqueeze(1)), dim=-1) # shape [batch, batch]

def strong_convex_func(x, lamb, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / lamb / 2.
    else:
        func = torch.exp(x / lamb) / torch.exp(torch.ones(1)) * lamb
    return func

def strong_convex_func_normalized(x, lamb, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / 2.
    else:
        func = torch.exp(x / lamb) / torch.exp(torch.ones(1))
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