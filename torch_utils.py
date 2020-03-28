import torch


def lstq(Y, A, lamb=0.0):
  """
  Differentiable least square
  :param A: m x n
  :param Y: n x 1
  """
  # Assuming A to be full column rank
  cols = A.shape[1]
  if cols == torch.matrix_rank(A):
    q, r = torch.qr(A)
    x = torch.inverse(r) @ q.T @ Y
  else:
    A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
    Y_dash = A.permute(1, 0) @ Y
    x = lstq(A_dash, Y_dash)
  return x


def t(a):
  """Convert to torch tensor"""
  return torch.FloatTensor(a)


def dt(a):
  """Convert to torch tensor"""
  return torch.DoubleTensor(a)


def npy(tensor):
  return tensor.detach().numpy()
