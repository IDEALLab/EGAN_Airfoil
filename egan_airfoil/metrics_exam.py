import torch
import numpy as np
import os
from models.cmpnts import BezierGenerator
from train_e import read_configs
from utils.dataloader import NoiseGenerator
from utils.metrics import ci_cons, ci_mll, ci_rsmth, ci_rdiv, ci_mmd

def load_generator(gen_cfg, save_dir, checkpoint, device='cpu'):
    ckp = torch.load(os.path.join(save_dir, checkpoint))
    generator = BezierGenerator(**gen_cfg).to(device)
    generator.load_state_dict(ckp['generator'])
    generator.eval()
    return generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/smm'
    X = np.load('../data/train.npy')
    X_test = np.load('../data/test.npy')
    _, gen_cfg, _, cz = read_configs('modified')
    generator = load_generator(gen_cfg, save_dir, 'modified499.tar', device=device)

    def gen_func(latent, noise=None):
        if isinstance(latent, int):
            N = latent
            input = NoiseGenerator(N, cz, device=device)()
        else:
            N = latent.shape[0]
            if noise is None:
                noise = np.zeros((N, cz[1]))
            input = torch.tensor(np.hstack([latent, noise]), device=device, dtype=torch.float)
        return generator(input)[0].cpu().detach().numpy().transpose([0, 2, 1]).squeeze()
    
    n_run = 10

    print("MLL: {} ± {}".format(*ci_mll(n_run, gen_func, X_test)))
    print("LSC: {} ± {}".format(*ci_cons(n_run, gen_func, cz[0])))
    print("RVOD: {} ± {}".format(*ci_rsmth(n_run, gen_func, X_test)))
    print("Diversity: {} ± {}".format(*ci_rdiv(n_run, X, gen_func)))
    print("MMD: {} ± {}".format(*ci_mmd(n_run, gen_func, X_test)))