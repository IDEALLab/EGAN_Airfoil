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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = 32
    epochs = 200

    dis_cfg, gen_cfg, egan_cfg, cz = read_configs('default')
    data_fname = '../data/airfoil_interp.npy'
    save_dir = '../saves/airfoil_dup'
    os.makedirs(save_dir, exist_ok=True)
    save_iter_list = list(np.linspace(1, 10, dtype=int) * 20 - 1)

    # build entropic gan on the device specified
    egan = assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, save_dir, device=device)

    # build dataloader and noise generator on the device specified
    dataloader = DataLoader(UIUCAirfoilDataset(data_fname, device=device), batch_size=batch, shuffle=True)
    noise_gen = NoiseGenerator(batch, sizes=cz, device=device)

    # build tensorboard summary writer
    writer = SummaryWriter(
        os.path.join(
            save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S')
            )
        )

    egan.train(
        epochs=epochs,
        num_iter_D=5, 
        num_iter_G=1,
        dataloader=dataloader, 
        noise_gen=noise_gen, 
        tb_writer=writer,
        report_interval=1,
        save_iter_list=save_iter_list
        )