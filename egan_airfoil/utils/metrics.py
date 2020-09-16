"""
Metrics compatible with PyTorch.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def mean_err(array):
    mean = np.mean(array)
    err = 1.96 * np.std(array) / len(array) ** .5
    return mean, err

def gaussian_kernel(X, Y, sigma=1.0):
    beta = 1. / (2. * sigma ** 2)
    dist = F.pairwise_distances(X, Y)
    s = beta * dist
    return torch.exp(-s)

def maximum_mean_discrepancy(X, Y):
    X = X.reshape((len(X), -1))
    Y = Y.reshape((len(Y), -1))
    mmd = gaussian_kernel(X, X).mean() \
        - 2 * gaussian_kernel(X, Y).mean() \
        + gaussian_kernel(Y, Y).mean()
    return mmd
    
def ci_mmd(n, gen_func, X_test):
    mmds = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(2000)
        mmds[i] = maximum_mean_discrepancy(X_gen, X_test).numpy()
    mean, err = mean_err(mmds)
    return mean, err


def sample_line(m, d, bounds):
    # Sample m points along a line parallel to a d-dimensional space's basis
    basis = np.random.choice(d)
    c = torch.rand(d).expand(m, d) # sample an arbitrary random vector
    c[:, basis] = torch.linspace(0.0, 1.0, m) # sample points along one direction of that random vector
    c = bounds[0] + (bounds[1] - bounds[0]) * c
    return c

def consistency(gen_func, latent_dim, bounds):
    n_eval = 100 # number of lines to be evaluated
    n_points = 50 #number of points sampled on each line
    mean_cor = 0
    for i in range(n_eval):
        c = sample_line(n_points, latent_dim, bounds)
        X = gen_func(c).reshape((n_points, -1))
        dist_c = torch.norm(c - c[0], dim=1).numpy()
        dist_X = torch.norm(X - X[0], dim=1).numpy()
        mean_cor += np.corrcoef(dist_c, dist_X)[0,1]
    return mean_cor / n_eval

def ci_cons(n, gen_func, latent_dim=2, bounds=(0.0, 1.0)):
    conss = np.zeros(n)
    for i in range(n):
        conss[i] = consistency(gen_func, latent_dim, bounds)
    mean, err = mean_err(conss)
    return mean, err