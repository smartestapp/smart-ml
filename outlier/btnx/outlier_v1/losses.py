import torch
import torch.nn.functional as F


def reconstruction_loss(x, x_hat):
    return F.mse_loss(input=x_hat, target=x)


def kl_divergence(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
