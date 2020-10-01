import torch
import numpy as np
import os, json

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.cmpnts import OTInfoDiscriminator1D, BezierGenerator
from models.gans import BezierEGAN
from utils.dataloader import UIUCAirfoilDataset, NoiseGenerator

def read_configs(name):
    with open(os.path.join('configs', name+'.json')) as f:
        configs = json.load(f)
        dis_cfg = configs['dis']
        gen_cfg = configs['gen']
        egan_cfg = configs['egan']
        cz = configs['cz']
    return dis_cfg, gen_cfg, egan_cfg, cz

def assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, save_dir, device='cpu'):
    discriminator = OTInfoDiscriminator1D(**dis_cfg).to(device)
    generator = BezierGenerator(**gen_cfg).to(device)
    egan = BezierEGAN(generator, discriminator, **egan_cfg, save_dir=save_dir)
    return egan

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dis_cfg, gen_cfg, egan_cfg, cz = read_configs('default')
    data_fname = '../data/airfoil_interp.npy'
    save_dir = '../saves/airfoil_dup'
    os.makedirs(save_dir, exist_ok=True)
    egan = assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, save_dir, device=device)

    batch = 32
    epochs = 100
    save_iter_list = list(np.linspace(1, 20, dtype=int) * 500 - 1)

    dataloader = DataLoader(UIUCAirfoilDataset(data_fname, device=device), batch)
    noise_gen = NoiseGenerator(batch, sizes=cz, device=device)
    
    writer = SummaryWriter(
        os.path.join(
            save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S')
            )
        )

    egan.train(
        epochs=epochs,
        dataloader=dataloader, 
        noise_gen=noise_gen, 
        tb_writer=writer,
        report_interval=1,
        save_iter_list=save_iter_list
        )