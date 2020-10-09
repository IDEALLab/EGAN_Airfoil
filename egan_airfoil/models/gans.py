import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import strong_convex_func, cross_distance, first_element

_eps = 1e-7

class GAN:
    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, 
        opt_g_lr: float=1e-4, opt_g_betas: tuple=(0.5, 0.99),
        opt_d_lr: float=1e-4, opt_d_betas: tuple=(0.5, 0.99),
        name: str=None, save_dir: str=None, 
        checkpoint: str=None
        ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.name = name
        self.save_dir = save_dir
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=opt_g_lr, betas=opt_g_betas)
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=opt_d_lr, betas=opt_d_betas)
        if checkpoint:
            self.load(os.path.join(save_dir, checkpoint))
    
    def loss_G(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.js_loss_G(batch, fake)
    
    def loss_D(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.js_loss_D(batch, fake)
    
    def js_loss_D(self, real, fake):
        return F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(real)), 
            torch.ones(len(real), 1, device=real.device)
            ) + F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(fake)), 
            torch.zeros(len(fake), 1, device=fake.device)
            )
    
    def js_loss_G(self, real, fake):
        return F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(fake)), 
            torch.ones(len(fake), 1, device=fake.device)
            )

    def _batch_hook(self, i, batch, noise_gen, tb_writer, **kwargs): pass
    
    def _batch_report(self, i, batch, noise_gen, tb_writer, **kwargs): pass

    def _epoch_hook(self, epoch, epochs, noise_gen, tb_writer, **kwargs): pass

    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('D Loss', loss_D(batch, noise_gen), epoch)
                tb_writer.add_scalar('G Loss', loss_G(batch, noise_gen), epoch)
            else:
                print('[Epoch {}/{}] D loss: {:d}, G loss: {:d}'.format(
                    epoch, epochs, loss_D(batch, noise_gen), loss_G(batch, noise_gen)))
    
    def _train_gen_criterion(self, batch, noise_gen, epoch): return True

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
        self.discriminator.train(); self.generator.train()
        self.optimizer_D.load_state_dict(ckp['optimizer_D'])
        self.optimizer_G.load_state_dict(ckp['optimizer_G'])
    
    def generate(self, noise, additional_info=False):
        if additional_info:
            return self.generator(noise)
        else: 
            return first_element(self.generator(noise))

    def train(
        self, dataloader, noise_gen, epochs, num_iter_D=5, num_iter_G=1, report_interval=5,
        save_iter_list=[100,], tb_writer=None, **kwargs
        ):
        for epoch in range(epochs):
            self._epoch_hook(epoch, epochs, noise_gen, tb_writer, **kwargs)
            for i, batch in enumerate(dataloader):
                self._batch_hook(i, batch, noise_gen, tb_writer, **kwargs)
                for _ in range(num_iter_D):
                    self.optimizer_D.zero_grad()
                    self.loss_D(batch, noise_gen, **kwargs).backward()
                    self.optimizer_D.step()
                if not self._train_gen_criterion(batch, noise_gen, epoch): continue
                for _ in range(num_iter_G):
                    self.optimizer_G.zero_grad()
                    self.loss_G(batch, noise_gen, **kwargs).backward()
                    self.optimizer_G.step()
                self._batch_report(i, batch, noise_gen, tb_writer, **kwargs)
            self._epoch_report(epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs)

            if save_iter_list and epoch in save_iter_list:
                self.save(epoch=epoch, noise=noise_gen)

class InfoGAN(GAN):
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        js_loss = self.js_loss_G(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        js_loss = self.js_loss_D(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def info_loss(self, fake, latent_code):
        q = self.discriminator(fake)[1]
        q_mean = q[:, :, 0]
        q_logstd = q[:, :, 1]
        epsilon = (latent_code - q_mean) / (torch.exp(q_logstd) + _eps)
        return torch.mean(q_logstd + 0.5 * epsilon ** 2)

class EGAN(GAN):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, lamb: float, 
        opt_g_lr: float=1e-4, opt_g_betas: tuple=(0.5, 0.99),
        opt_d_lr: float=1e-4, opt_d_betas: tuple=(0.5, 0.99),
        name: str=None, save_dir: str=None, 
        checkpoint: str=None
        ):
        super().__init__(
            generator, discriminator, 
            opt_g_lr=opt_g_lr, opt_g_betas=opt_g_betas,
            opt_d_lr=opt_d_lr, opt_d_betas=opt_d_betas,
            name=name, save_dir=save_dir, checkpoint=checkpoint
            )
        self.lamb = lamb
        # self._lamb = lamb

    def loss_D(self, batch, noise_gen, **kwargs):
        return -self.loss_G(batch, noise_gen)
    
    def loss_G(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.dual_loss(batch, fake)

    def dual_loss(self, real, fake):
        v, d_r, d_f = self._cal_v(real, fake) 
        smooth = strong_convex_func(v, lamb=self.lamb).mean()
        return d_r.mean() - d_f.mean() - smooth

    def surrogate_ll(self, test, noise_gen):
        noise, p_x = noise_gen(output_prob=True); fake = self.generate(noise)
        prob = self._estimate_prob(test, fake, p_x) # [batch, n_test]
        ep_dist = (cross_distance(test, fake) * prob).sum(dim=0) # [n_test]
        entropy = (-prob * torch.log(prob + _eps)).sum(dim=0) # [n_test]
        ep_lpx = (torch.log(p_x) * prob).sum(dim=0) # [n_test]
        return -ep_dist / self.lamb + entropy + ep_lpx # [n_test] log likelihood surrogate 2.7
    
    def _cal_v(self, real, fake):
        d_r = first_element(self.discriminator(real))[:, 0] # [r_batch, 1]
        d_f = first_element(self.discriminator(fake))[:, 1] # [f_batch, 1]
        v = torch.squeeze(d_r.unsqueeze(0) - d_f.unsqueeze(1), dim=-1) \
            - cross_distance(real, fake) # [f_batch, r_batch] v(y,^y) grid. Should add one argument to choose distance functions.
        return v, d_r, d_f

    def _estimate_prob(self, test, fake, p_x):
        v_star, _, _ = self._cal_v(test, fake) # [batch, n_test] term inside exp() of 4.3. 
        exp_v = torch.exp((v_star - v_star.max(dim=0)) / self.lamb) # [batch, n_test] avoid numerical instability.
        prob = p_x * exp_v # [batch, n_test] unnormalized 4.3
        return prob / prob.sum(dim=0) # [batch, n_test] normalize over x to obtain 4.3 P(x|y)
    
    def _train_gen_criterion(self, batch, noise_gen, epoch):
        _, d_r, d_f = self._cal_v(batch, self.generate(noise_gen()))
        return d_r.mean() - d_f.mean() > 0

    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('Dual Loss', loss_G(batch, noise_gen), epoch)
            else:
                print('[Epoch {}/{}] Dual loss: {:d}'.format(
                    epoch, epochs, loss_G(batch, noise_gen)))

class InfoEGAN(EGAN, InfoGAN):
    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return -dual_loss + info_loss
    
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return dual_loss + info_loss

class BezierEGAN(InfoEGAN):
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return dual_loss + info_loss + 10 * reg_loss
    
    def regularizer(self, cp, w, pv, intvls):
        w_loss = torch.mean(w[:, :, 1:-1])
        cp_loss = torch.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
        end_loss = torch.pairwise_distance(cp[:, :, 0], cp[:, :, -1]).mean()
        reg_loss = w_loss + cp_loss + end_loss
        return reg_loss
    
    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
            fake, cp, w, pv, intvls = self.generator(noise)
            v, d_r, d_f = self._cal_v(batch, fake)
            smooth = strong_convex_func(v, lamb=self.lamb).mean()
            info_loss = self.info_loss(fake, latent_code)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            if tb_writer:
                # tb_writer.add_scalar('Dual Loss', dual_loss, epoch)
                tb_writer.add_scalars('Dual Loss', {
                                        'dual': d_r.mean() - d_f.mean() - smooth,
                                        'emd': d_r.mean() - d_f.mean(),
                                        'smooth': smooth
                                        }, epoch)
                tb_writer.add_scalar('Info Loss', info_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)
            else:
                print('[Epoch {}/{}] Dual loss: {:d}, Info loss: {:d}, Regularization loss: {:d}'.format(
                    epoch, epochs,  d_r.mean() - d_f.mean() - smooth, info_loss, reg_loss))
            try: kwargs['plotting'](epoch, fake) 

class BezierGAN(InfoGAN):
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        js_loss = self.js_loss_G(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return js_loss + info_loss + 10 * reg_loss

    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        js_loss = self.js_loss_D(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss
    
    def regularizer(self, cp, w, pv, intvls):
        w_loss = torch.mean(w[:, :, 1:-1])
        cp_loss = torch.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
        end_loss = torch.pairwise_distance(cp[:, :, 0], cp[:, :, -1]).mean()
        reg_loss = w_loss + cp_loss + end_loss
        return reg_loss
    
    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
            fake, cp, w, pv, intvls = self.generator(noise)
            js_loss = self.js_loss_D(batch, fake)
            info_loss = self.info_loss(fake, latent_code)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            if tb_writer:
                tb_writer.add_scalar('JS Loss', js_loss, epoch)
                tb_writer.add_scalar('Info Loss', info_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)
            else:
                print('[Epoch {}/{}] JS loss: {:d}, Info loss: {:d}, Regularization loss: {:d}'.format(
                    epoch, epochs,  js_loss, info_loss, reg_loss))
            try: kwargs['plotting'](epoch, fake) 