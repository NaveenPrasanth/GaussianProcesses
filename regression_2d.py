import matplotlib.pyplot as plt
from torch.optim import Adam
import torch
import numpy as np

from gp import GP


if __name__ == '__main__':
  gp = GP()
  f = lambda x: ( torch.cos(2 * x[:, 0]) + torch.sin(x[:, 1]) ).view(-1)
  n1, n2, ny = 100, 100, 5
  domain = (-5, 5)
  X_data = torch.distributions.Uniform(
      domain[0] + 2, domain[1] - 2).sample((n1, 2))
  y_data = f(X_data)
  X_test = torch.linspace(domain[0], domain[1], n2)
  X_test = torch.stack([X_test, X_test]).view(-1, 2)
  y_test = f(X_test)

  gp.fit(X_data, y_data, True, epochs=10000)
  y_pred, _ = gp.predict(X_test)
  print('MSE : ', ((y_pred - y_test)**2).mean().item() )
