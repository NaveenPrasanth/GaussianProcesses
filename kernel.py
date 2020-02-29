import torch


def rbf(Xa, Xb, variance, lengthscale):
  sqdist = (Xa**2).sum(1).view(-1, 1) + (Xb**2).sum(1) - 2 * torch.mm(Xa, Xb.T)
  return variance**2 * torch.exp((-0.5 * sqdist) / (lengthscale ** 2))
