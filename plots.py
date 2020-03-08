import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('darkgrid')
np.random.seed(69)


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


def illustrate_covariance_matrix(cov):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))
    im = ax1.imshow(cov, cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    # cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title(('Covariance Matrix')) 
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ax1.grid(False)
    # render figure
    fig.tight_layout()
    plt.show()


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


def posterior_plot(X1, y1, X2, mu, sigma):
    """Plot the postior distribution and some samples"""
    fig, ax1 = plt.subplots(
      nrows=1, ncols=1, figsize=(6, 6))
    # Plot the distribution of the function (mean, covariance)
    # ax1.fill_between(X2.flat,
    #     mu - 2 * sigma, mu + 2 * sigma,
    #     color='red', alpha=0.15, label='$2 \sigma_{2|1}$')
    ax1.plot(X2, mu, 'r-', lw=2, label='$\mu_{2|1}$')

    # scatter training points
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')

    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.axis([X2.min() - 1, X2.max() + 1, mu.min()-1, mu.max()+1])
    ax1.legend()

    plt.tight_layout()
    plt.show()


def plot_gp(x, y, y_pred, sigma):
      fig = plt.figure(figsize=(8, 6))
      plt.xticks(np.arange(-12, max(x)+1, 6.0))
      plt.xlim([-12.01, 12.01])
      plt.ylim([-0.4, 1.2])
      plt.xlabel('x', fontsize=23)
      plt.plot(x, y, c='red', alpha=0.9)
      plt.scatter(x, y_pred, s=4, c='#854EC0', alpha=0.5)
      plt.fill(np.concatenate([x, x[::-1]]),
               np.concatenate([y_pred - 1.9600 * sigma,
                              (y_pred + 1.9600 * sigma)[::-1]]),
               alpha=.3, fc='#ED358E', ec='None', label='95% confidence interval')
      plt.ylabel('$\psi$', fontsize=20)
      plt.title('Wave Function', fontsize=20)
      plt.show()

