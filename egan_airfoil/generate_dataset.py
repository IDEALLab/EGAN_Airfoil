import torch
import numpy as np
import os
from models.cmpnts import BezierGenerator
from train_v import read_configs
from plot_latent import load_generator
from utils.shape_plot import plot_samples, plot_grid
from utils.dataloader import NoiseGenerator


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/airfoil_dup_v_2'
    _, gen_cfg, _, cz = read_configs('vanilla')

    epoch = 550
    noise = NoiseGenerator(3000, sizes=cz, device=device)()

    generator = load_generator(gen_cfg, save_dir, 'vanilla{}.tar'.format(epoch-1), device=device)

    dp, cp, w, _, _ = generator(noise)
    print(dp.shape, cp.shape, w.shape)

    dp = dp.cpu().detach().numpy().transpose([0, 1, 2])
    cp = cp.cpu().detach().numpy()
    w = w.cpu().detach().numpy()
    cpw = np.concatenate([cp, w], axis=1)
    print(cpw.shape)

    np.save('fake_airfoils.npy', dp)
    # np.save('fake_ctrl_points.npy', cp)
    # np.save('fake_weights.npy', w)
    np.save('fake_cpws.npy', cpw)
