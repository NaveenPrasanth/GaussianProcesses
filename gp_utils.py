import torch


def jit_op(op, x, max_tries=10):
  jitter_size = x.diag().mean()
  try:
    return op(x)
  except Exception as e:
    pass

  for i in range(max_tries):
    try:
      _jitter = 10. ** (-max_tries + i) * torch.eye(*x.shape, dtype=x.dtype)
      return op(x + _jitter)
    except RuntimeError as e:
      pass

  raise RuntimeError('Max tries exceeded!')


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
    #if Y_dash.dim() == 1:
    #  Y_dash = Y_dash.view(-1, 1)
    x = lstq(Y_dash, A_dash)
  return x


def cholesky(x):
  return jit_op(torch.cholesky, x)
