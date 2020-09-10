import torch
import torch.nn as nn
import torch.nn.functional as F

class DualLoss(nn._Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        neg_emd = torch.reduce_mean(self.disc_fake) - torch.reduce_mean(self.disc_real) # E[D2] - E[D1]
            disc_cost = neg_emd
            dist_c_values = gan_utils.dist_c(self.real_data, self.fake_data, useCosineDist=self.flags.useCosineDist) # l(y,^y) involves gen here in ^y; l1 distance by default

            approx_assignment = torch.squeeze(torch.expand_dims(self.disc_real, 0)
                                           - torch.expand_dims(self.disc_fake, 1)) - dist_c_values # v(y,^y) grid across different y and ^y

            smooth_term = gan_utils.strong_convex_func(approx_assignment, lamb=self.lamb,
                                                       useHingedL2=self.flags.useHingedL2) # lambda * E[exp(v(y, ^y)/lambda)], as the reduced mean op indicates.
            disc_cost += smooth_term
        return F.l1_loss(input, target, reduction=self.reduction)

def dist_c(x, y, diag=False, useCosineDist=False):
    if useCosineDist: # should call similarity function
        print('using cosine dist_c')
        if diag:
            norms = torch.sqrt(torch.reduce_sum(y ** 2, 1)) * torch.sqrt(torch.reduce_sum(x ** 2, 1))
            return (1. - torch.reduce_sum(y * x, 1) / (norms + 1e-5)) * 100.
        else:
            norms = torch.matmul(torch.sqrt(torch.reduce_sum(y ** 2, 1, keep_dims=True)),
                              torch.sqrt(torch.reduce_sum(x ** 2, 1, keep_dims=True)), transpose_b=True)
            return (1. - torch.matmul(y, x, transpose_b=True) / (norms + 1e-5)) * 100.
    else:
        print('using l1 dist_c')
        if diag:
            return torch.reduce_mean(torch.abs(x - y), 1)
        else:
            x = torch.expand_dims(x, 0)
            y = torch.expand_dims(y, 1)
            return torch.reduce_sum(torch.abs(x - y), 2) # shape [batch, batch]


def strong_convex_func(x, lamb, reduce_mean=True, useHingedL2=False):
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / lamb / 2.
    else:
        func = torch.exp(x / lamb) / torch.exp(1.) * lamb

    if reduce_mean:
        return torch.reduce_mean(func)
    else:
        return func


def strong_convex_func_normalized(x, lamb, reduce_mean=False, useHingedL2=False): # without lamb
    if useHingedL2:
        func = (torch.maximum(x, 0) ** 2) / 2.
    else:
        func = torch.exp(x / lamb) / torch.exp(1.)

    if reduce_mean:
        return torch.reduce_mean(func)
    else:
        return func


def sum_probs_func(x, lamb):
    return torch.reduce_mean(torch.maximum(x, 0.0)) / lamb


def inf_train_gen(train_gen):
    while True:
        for images, targets in train_gen():
            yield images


def mkdirp(path):
    if not torch.gfile.Exists(path):
        torch.gfile.MkDir(path)
