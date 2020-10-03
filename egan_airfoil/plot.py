import torch
import numpy as np
import os
from models.cmpnts import BezierGenerator
from train import read_configs
from utils.shape_plot import plot_samples, plot_grid
from utils.dataloader import NoiseGenerator

def load_generator(gen_cfg, save_dir, checkpoint, device='cpu'):
    ckp = torch.load(os.path.join(save_dir, checkpoint))
    generator = BezierGenerator(**gen_cfg).to(device)
    generator.load_state_dict(ckp['generator'])
    generator.eval()
    return generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/airfoil_dup'
    _, gen_cfg, _, cz = read_configs('default')

    epoch = 200

    generator = load_generator(gen_cfg, save_dir, 'default{}.tar'.format(epoch-1), device=device)
    noise_gen = NoiseGenerator(36, sizes=cz, device=device)

    samples = generator(noise_gen())[0].cpu().detach().numpy().transpose([0, 2, 1])
    plot_samples(None, samples, scale=1.0, scatter=False, symm_axis=None, lw=1.2, alpha=.7, c='k', fname='epoch {}'.format(epoch))