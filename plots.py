import numpy as np
import matplotlib.pyplot as plt
import math

from monet import *


def illustrate_kernel(k):
    """Illustrate covariance matrix and function
    
    k : kernel function
    """

    # Show covariance matrix example from exponentiated quadratic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    xlim = (-3, 3)
    X = np.expand_dims(np.linspace(*xlim, 25), 1)
    cov = k(X, X)
    # Plot covariance matrix
    im = ax1.imshow(cov, cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title((
        'Exponentiated quadratic \n'
        'example of covariance matrix'))
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ticks = list(range(xlim[0], xlim[1]+1))
    ax1.set_xticks(np.linspace(0, len(X)-1, len(ticks)))
    ax1.set_yticks(np.linspace(0, len(X)-1, len(ticks)))
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.grid(False)

    # Show covariance with X=0
    xlim = (-4, 4)
    X = np.expand_dims(np.linspace(*xlim, num=100), 1)
    zero = np.array([[0]])
    covΣ0 = k(X, zero)
    # Make the plots
    ax2.plot(X[:,0], covΣ0[:,0], label='$k(x,0)$')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('covariance', fontsize=13)
    ax2.set_title((
        'covariance\n'
        'between $x$ and $0$'))
    # ax2.set_ylim([0, 1.1])
    ax2.set_xlim(*xlim)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()


def illustrate_covariance_matrix(cov, ax=None, include_plot=True,
    xlabel='$x$', ylabel='$x$', title='Covariance Matrix'):
  if include_plot:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
  im = ax.imshow(cov, cmap='cool')
  #cbar = plt.colorbar(
  #    im, ax=ax, fraction=0.045, pad=0.05)
  # cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
  ax.set_title(title, fontsize=13)
  ax.set_xlabel(xlabel, fontsize=13)
  ax.set_ylabel(ylabel, fontsize=13)
  ax.grid(False)
  if include_plot:
    # render figure
    fig.tight_layout()
    plt.show()

  return ax


def illustrate_samples(X, ys):
    plt.figure(figsize=(6, 4))
    for i in range(ys.shape[0]):
      plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title((
      'Different function realizations at {} points\n'.format(len(X)) +\
      'sampled from a Gaussian process'))
    plt.xlim([X.min(), X.max()])
    plt.show()


def just_plot(x, y):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title(('Just a plot, man.. geez'))
    plt.xlim([x.min(), x.max()])
    plt.show()


def just_scatter(x, y):
    plt.figure(figsize=(7, 4))
    plt.scatter(x, y)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title(('Just a plot, man.. geez'))
    plt.xlim([x.min() - 1, x.max() + 1])
    plt.show()


def plot_gp(mu, sigma, x_data=None, y_data=None, x_test=None, y_test=None, 
    ax=None, xlabel='$x$', ylabel='$y$', num_x_samples=30,
    include_plot=True):
  if ax is None:
    fig = plt.figure(figsize=(7, 6))
    ax = plt.axes()
  ax.set_xlabel(xlabel, fontsize=13)
  if x_data is not None and y_data is not None:
    ax.scatter(x_data, y_data, s=35, c=colors[2], alpha=0.9, zorder=3)
  if x_test is None:
    x_test = np.linspace(min(x_data), max(x_data), num_x_samples)
  idx = np.arange(0, x_test.shape[0], x_test.shape[0] // num_x_samples)
  if y_test is not None:
    ax.plot(x_test, y_test, c=colors[2], alpha=0.9, linestyle='dashed', zorder=1)
  ax.scatter(x_test[idx], mu[idx], s=25, c=colors[0], alpha=0.9, zorder=2)
  ax.plot(x_test, mu, c=colors[0], alpha=0.9, linestyle='dashed')
  ax.fill(np.concatenate([x_test, x_test[::-1]]),
           np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
           alpha=.3, fc=colors[0], ec='None', label='95% confidence interval')
  ax.set_ylabel(ylabel, fontsize=13)
  ax.set_title('Wave Function', fontsize=13)
  ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
  ax.set_ylim(min(mu) - 1, max(mu) + 1)
  if y_test is not None:
    ax.set_ylim(min(y_test) - 1, max(y_test) + 1)

  if include_plot:
    plt.show()

  return ax


def collate_plots(mu, x_test,
    cov=None, K=None, x_data=None, y_data=None, y_test=None,
    pwidth=4., pheight=3.):

  nplots, plot_fns = 0, []
  # resolve sigma
  if cov is not None:
    # get sigma from cov
    diag = cov.diagonal()
    sigma = np.sqrt(np.where(diag > 0., diag, np.zeros_like(diag)))

  # get dimensions of x
  xdim = x_test.reshape(x_test.shape[0], -1).shape[1]
  nplots += xdim
  for i in range(xdim):
    x_data_ = x_data.reshape(x_data.shape[0], -1)[:, i].reshape(-1)
    x_test_ = x_test.reshape(x_test.shape[0], -1)[:, i].reshape(-1)
    plot_fns.append( lambda ax : plot_gp(
      mu, sigma, x_data=x_data_, y_data=y_data, x_test=x_test_, y_test=y_test,
      ax=ax, include_plot=False))

  # covariance plots
  if K is not None:
    nplots += 1
    plot_fns.append(lambda ax : illustrate_covariance_matrix(
      K, ax, include_plot=False, title='$k(x, x)$'))
  if cov is not None:
    nplots += 1
    plot_fns.append(lambda ax : illustrate_covariance_matrix(
      cov, ax, include_plot=False, title='$\Sigma_{2|1}$'))

  ncols = min(4, nplots)
  nrows = math.ceil(nplots / ncols)
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
      figsize=(3 + pwidth * ncols, 2 + pheight * nrows)) 
  axes = axes.reshape(-1)
  # render plots
  for i in range(len(plot_fns)):
    plot_fns[i](axes[i])
  plt.show()
