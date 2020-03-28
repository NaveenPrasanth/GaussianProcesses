
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp import GP


class GPSelect:

  def __init__(self, components=1, name="sklearn"):
    if name == "sklearn":
      kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
      if components == 2:
        kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5, 5, 5], (1e-2, 1e2))
      self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

    elif name == 'torchgp':
      self.gp = GP()


