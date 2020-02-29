import matplotlib.pyplot as plt
from torch.optim import Adam
import torch
import numpy as np

from gp import GP


if __name__ == '__main__':
  gp = GP()
  f = lambda x: ( 0.4 * x + torch.cos(2 * x) + torch.sin(x)).view(-1)
  n1, n2, ny = 20, 100, 5
  domain = (-5, 5)
  X_data = torch.distributions.Uniform(
      domain[0] + 2, domain[1] - 2).sample((n1, 1))
  y_data = f(X_data)
  X_test = torch.linspace(
          domain[0], domain[1], n2).view(-1, 1)

  gp.fit(X_data, y_data, True)
  gp.plot()
