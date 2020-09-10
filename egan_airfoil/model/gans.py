import os
import torch
import torch.nn as nn
from cmpnts import BezierGenerator, OTInfoDiscriminator1D
from utils import DualLoss

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
    
    def loss_G(self, batch):
        loss = None
        return loss
    
    def loss_D(self, batch):
        loss = None
        return loss
    
    def _report(self, epoch, epochs, batch):
        print('[Epoch {}/{}] D loss: {:d}, G loss: {:d}'.format(
            epoch, epochs, loss_D(batch), loss_G(batch)))

    def save(self, **kwargs):
        torch.save({
            'records': kwargs,
            'discriminator': self.discriminator.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict()
            }, os.path.join(self.save_dir, self.name+str(kwargs['epoch'])+'.tar'))

    def load(self, checkpoint):
        ckp = torch.load(os.path.join(self.save_dir, checkpoint))
        self.discriminator.load_state_dict(ckp['discriminator'])
        self.discriminator.eval()
        self.generator.load_state_dict(ckp['generator'])
        self.generator.eval()
        self.optimizer_D.load_state_dict(ckp['optimizer_D'])
        self.optimizer_G.load_state_dict(ckp['optimizer_G'])

    def train(
        self, dataloader, epochs, num_iter_D=5, num_iter_G=1, report_interval=5,
        save_iter_list=[100,]
        ):
        for epoch in range(epochs):
            for i, batch in enumerate(dataloader):
                # Train discriminator
                for _ in range(num_iter_D):
                    self.optimizer_D.zero_grad()
                    self.loss_D(batch).backward()
                    self.optimizer_D.step()
                # Train generator
                for _ in range(num_iter_G):
                    self.optimizer_G.zero_grad()
                    self.loss_G(batch).backward()
                    self.optimizer_G.step()
            
            if epoch % report_interval == 0:
                self._report(epoch, epochs, batch)

            if save_iter_list and epoch in save_iter_list:
                self.save(epoch=epoch)


class EGAN(GAN):
    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, 
        name: str, save_dir: str, checkpoint: str=None
        ):
        super().__init__(generator, discriminator, name, save_dir, checkpoint)
    
    def loss_D(self, batch):
        return super().loss_D(batch)
    
    def loss_G(self, batch):
        return super().loss_G(batch)
    
    def _report(self, epoch, epochs, batch):
        return super()._report(epoch, epochs, batch)