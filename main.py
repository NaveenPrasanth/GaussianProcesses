import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam
import numpy as np

from gaussian_processes import GP


if __name__ == '__main__':
  gp = GP()
  f = lambda x: torch.sinh(x).view(-1)
  n1, n2, ny = 4, 300, 5
  domain = (-5, 5)
  X_data = torch.distributions.Uniform(
      domain[0] + 2, domain[1] - 2).sample((n1, 1))
  y_data = f(X_data)
  X_test = torch.linspace(
          domain[0], domain[1], n2).view(-1, 1)

  gp.fit(X_data, y_data, True)
  gp.plot()
