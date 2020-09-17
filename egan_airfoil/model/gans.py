import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmpnts import BezierGenerator, OTInfoDiscriminator1D
from utils import strong_convex_func, cross_distance, first_element

_eps = 1e-7

class GAN:
    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, 
        name: str, save_dir: str, checkpoint: str=None
        ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.name = name
        self.save_dir = save_dir
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.99))
        self.optimizer_D = torch.optim.Adam(
            self.critics.parameters(), lr=2e-4, betas=(0.5, 0.99))
        if checkpoint:
            self.load(os.path.join(save_dir, checkpoint))
    
    def loss_G(self, batch, noise_gen, **kwargs):
        loss = None
        return loss
    
    def loss_D(self, batch, noise_gen, **kwargs):
        loss = None
        return loss
    
    def _batch_hook(self, i, batch, **kwargs): pass
    
    def _batch_report(self, i, batch, noise_gen, **kwargs): pass

    def _epoch_hook(self, epoch, epochs, **kwargs): pass

    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, **kwargs):
        if epoch % report_interval == 0:
            print('[Epoch {}/{}] D loss: {:d}, G loss: {:d}'.format(
                epoch, epochs, loss_D(batch, noise_gen), loss_G(batch, noise_gen)))

    def save(self, **kwargs):
        torch.save({
            'discriminator': self.discriminator.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'records': kwargs
            }, os.path.join(self.save_dir, self.name+str(kwargs['epoch'])+'.tar'))

    def load(self, checkpoint):
        ckp = torch.load(os.path.join(self.save_dir, checkpoint))
        self.discriminator.load_state_dict(ckp['discriminator'])
        self.generator.load_state_dict(ckp['generator'])
        self.optimizer_D.load_state_dict(ckp['optimizer_D'])
        self.optimizer_G.load_state_dict(ckp['optimizer_G'])
        self.discriminator.eval(); self.generator.eval()
    
    def generate(self, noise):
        return first_element(self.generator(noise))

    def train(
        self, dataloader, noise_gen, epochs, num_iter_D=5, num_iter_G=1, report_interval=5,
        save_iter_list=[100,], **kwargs
        ): # the noise generator should also be something like the dataloader.
        for epoch in range(epochs):
            self._epoch_hook(epoch, epochs, **kwargs)
            for i, batch in enumerate(dataloader):
                self._batch_hook(i, batch)
                # Train discriminator
                for _ in range(num_iter_D):
                    self.optimizer_D.zero_grad()
                    self.loss_D(batch, noise_gen, **kwargs).backward()
                    self.optimizer_D.step()
                # Train generator
                for _ in range(num_iter_G):
                    self.optimizer_G.zero_grad()
                    self.loss_G(batch, noise_gen, **kwargs).backward()
                    self.optimizer_G.step()
                self._batch_report(i, batch, noise_gen, **kwargs)
            self._epoch_report(epoch, epochs, batch, noise_gen, report_interval, **kwargs)

            if save_iter_list and epoch in save_iter_list:
                self.save(epoch=epoch, noise=noise_gen)

class EGAN(GAN):
    def __init__(self, generator, discriminator, lamb, name, save_dir, checkpoint=None):
        super().__init__(generator, discriminator, name, save_dir, checkpoint=checkpoint)
        self.lamb = lamb

    def loss_D(self, batch, noise_gen, **kwargs):
        return -self.loss_G(batch, noise_gen)
    
    def loss_G(self, batch, noise_gen, **kwargs):
        fake = first_element(self.generator(noise_gen()))
        loss = self.dual_loss(batch, fake)
        return loss
        
    def dual_loss(self, real, fake):
        d_r = first_element(self.discriminator(real))[:, 0]
        d_f = first_element(self.discriminator(fake))[:, 1]

        emd = torch.mean(d_r) - torch.mean(d_f) # E[D1] - E[D2]
        v = torch.squeeze(d_r.unsqueeze(0) - d_f.unsqueeze(1)) \
            - cross_distance(real, fake) # v(y,^y) grid across different y and ^y. Should add one argument to choose distance functions.
        smooth = strong_convex_func(v, lamb=self.lamb).mean()
        return emd - smooth
    
    def _estimate_probs(self, noise, fake, test):
        """
        Estimate P(x|y=y_test) in 4.3
        """
        # unnormalized density
        p_x = torch.exp(torch.diag(-0.5 * np.dot(noise, noise.T))) # [batch, 1] Gaussian P(x) unnormalized. 
        # Probability of x of every entry in a batch. Should be modified to be compatible with infogan.

        d_r = first_element(self.discriminator(test))[:, 0]
        d_f = first_element(self.discriminator(fake))[:, 1]
        v_star = (torch.squeeze(d_r.unsqueeze(0) - d_f.unsqueeze(1), dim=-1) \
            - cross_distance(test, fake)) / self.lamb # [batch, n_test] term inside exp() of 4.3. 
        exp_v = torch.exp(v_star - v_star.max(dim=0)) # [batch, n_test] avoid numerical instability.
        prob = p_x * exp_v # [batch, n_test] unnormalized 4.3
        return prob / prob.sum(dim=0) # [batch, n_test] normalize over x to obtain 4.3 P(x|y)

    def compute_LL(self, test, noise_gen):
        noise = noise_gen(); fake = self.generate(noise)
        prob = self._estimate_probs(noise, fake, test) # [batch, n_test] P(x|y)
        
        dist = cross_distance(test, fake) # [batch, n_test]
        e_dist = torch.sum(dist * prob, dim=0) # [n_test] expectation of dist
        ent = torch.sum(-prob * torch.log(prob + _eps), dim=0) # [n_test] entropy

        e_norm = torch.sum(-0.5 * (noise**2).sum(dim=1, keepdim=True) * prob, dim=0) # [n_test] expectation of norm. Should this be L2 for all dimension?
        # computing likelihood of the latent variable? Should be modified to accommodate uniformly distributed latent variables

        lower_bound = -1 / self.lamb * (e_dist - self.lamb * ent) + e_norm # [n_test] log likelihood lower bound 2.7  
        return lower_bound 
    
    def _epoch_hook(self, epoch, epochs, **kwargs):
        if epoch == 0:
            self.lamb = kwargs['lamb']

    def _epoch_report(self, epoch, epochs, batch, report_interval, **kwargs):
        return super()._report(epoch, epochs, batch, report_interval, **kwargs) # modify

class InfoEGAN(EGAN):
    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = first_element(self.generator(noise))
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return -dual_loss + info_loss
    
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = first_element(self.generator(noise))
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return dual_loss + info_loss
    
    def info_loss(self, fake, latent_code):
        return F.mse_loss(self.discriminator(fake)[1], latent_code)

class BezierEGAN(InfoEGAN):
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return dual_loss + info_loss + 10 * reg_loss
    
    def regularizer(self, cp, w, pv, intvls):
        w_loss = torch.mean(w[:, 1:-1])
        cp_loss = torch.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
        end_loss = torch.pairwise_distance(cp[:, :, 0], cp[:, :, -1]).mean()
        reg_loss = w_loss + cp_loss + end_loss
        return reg_loss
    
    def _epoch_report(self, epoch, epochs, batch, report_interval, **kwargs):
        return super()._report(epoch, epochs, batch, report_interval, **kwargs) # modify