import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from utils.dataloader import UIUCAirfoilDataset
from utils.shape_plot import plot_samples

if __name__ == '__main__':
    data_fname = '../data/airfoil_interp.npy'
    dataloader = DataLoader(UIUCAirfoilDataset(data_fname), batch_size=36, shuffle=True)

    for i in range(3):
        samples = next(iter(dataloader)).numpy().transpose([0, 2, 1])
        plot_samples(None, samples, scale=1.0, scatter=False, symm_axis=None, lw=1.2, alpha=.7, c='k', fname=str(i))