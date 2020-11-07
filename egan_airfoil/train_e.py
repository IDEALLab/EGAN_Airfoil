import torch
import numpy as np
import os, json

from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.cmpnts import InfoDiscriminator1D, BezierGenerator
from models.gans import BezierEGAN, BezierSEGAN
from utils.dataloader import UIUCAirfoilDataset, NoiseGenerator
from utils.shape_plot import plot_samples

def read_configs(name):
    with open(os.path.join('configs', name+'.json')) as f:
        configs = json.load(f)
        dis_cfg = configs['dis']
        gen_cfg = configs['gen']
        egan_cfg = configs['egan']
        cz = configs['cz']
    return dis_cfg, gen_cfg, egan_cfg, cz

def assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, save_dir, device='cpu'):
    discriminator = InfoDiscriminator1D(**dis_cfg).to(device)
    generator = BezierGenerator(**gen_cfg).to(device)
    egan = BezierSEGAN(generator, discriminator, **egan_cfg, save_dir=save_dir)
    return egan

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = 32
    epochs = 500
    save_intvl = 50

    dis_cfg, gen_cfg, egan_cfg, cz = read_configs('modified')
    # data_fname = '../data/airfoil_interp.npy'
    data_fname = '../data/train.npy'
    save_dir = '../saves/sinkhorn10'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'runs'), exist_ok=True)

    # X_train, X_test = train_test_split(np.load(data_fname), train_size=0.8, shuffle=True)
    # np.save(os.path.join(save_dir, 'train.npy'), X_train)
    # np.save(os.path.join(save_dir, 'test.npy'), X_test)
    X_train = np.load(data_fname)

    save_iter_list = list(np.linspace(1, epochs/save_intvl, dtype=int) * save_intvl - 1)

    # build entropic gan on the device specified
    egan = assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, save_dir, device=device)

    # build dataloader and noise generator on the device specified
    dataloader = DataLoader(UIUCAirfoilDataset(X_train, device=device), batch_size=batch, shuffle=True)
    noise_gen = NoiseGenerator(batch, sizes=cz, device=device)

    # build tensorboard summary writer
    tb_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(os.path.join(tb_dir, 'images'), exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    def epoch_plot(epoch, fake, *args, **kwargs):
        if (epoch + 1) % 10 == 0:
            samples = fake.cpu().detach().numpy().transpose([0, 2, 1])
            plot_samples(
                None, samples, scale=1.0, scatter=False, symm_axis=None, lw=1.2, alpha=.7, c='k', 
                fname=os.path.join(tb_dir, 'images', 'epoch {}'.format(epoch+1))
                )

    egan.train(
        epochs=epochs,
        num_iter_D=1, 
        num_iter_G=1,
        dataloader=dataloader, 
        noise_gen=noise_gen, 
        tb_writer=writer,
        report_interval=1,
        save_iter_list=save_iter_list,
        plotting=epoch_plot
        )