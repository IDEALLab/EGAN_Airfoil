import torch
import numpy as np
import os
from models.cmpnts import BezierGenerator
from train_v import read_configs
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

    save_dir = '../saves/airfoil_dup_v'
    _, gen_cfg, _, cz = read_configs('vanilla')

    epochs = [300,]
    points_per_dim = 10

    for epoch in epochs:
        generator = load_generator(gen_cfg, save_dir, 'vanilla{}.tar'.format(epoch-1), device=device)

        Z = np.array(np.meshgrid(np.linspace(0, 1, points_per_dim), np.linspace(0, 1, points_per_dim))).T.reshape(-1,2)
        for i in range(cz[0]):
            latent = np.insert(Z, i, 0.5, axis=1)
            
            noise = torch.tensor(
                np.hstack([
                    latent, 
                    np.random.randn(len(latent), cz[1])
                    ]), device=device, dtype=torch.float32)

            samples = generator(noise)[0].cpu().detach().numpy().transpose([0, 2, 1])
            plot_samples(Z, samples, scale=1.0, scatter=False, symm_axis=None, lw=1.2, alpha=.7, c='k', fname='epoch {} {}'.format(epoch, i))
