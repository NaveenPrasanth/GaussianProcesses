from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from monet import *
from torch_utils import *
import matplotlib.pyplot as plt
import numpy as np


class GPSelect:
    def __init__(self, components=1, name="sklearn"):
        if name == "sklearn":
            kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5]*components, (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

    def get_gp(self):
        return self.gp


class GP:
    def __init__(self, gp, data):
        self.gp = gp
        self.x_data = data.loc[:, list(data)[:-1]].to_numpy()
        self.y_data = data.loc[:, list(data)[-1]].to_numpy()

    def fit(self):
        self.gp.fit(t(self.x_data), t(self.y_data))

    def predict(self, X):
        y_pred, sigma = self.gp.predict(X, return_std=True)
        sigma[sigma < 0.] = 0.
        return y_pred, sigma

    def plot_data(self):
        number_of_plots = self.x_data.shape[1]
        for i in range(number_of_plots):
            self.__plot_gp(x_data=self.x_data[:, i], y_data=self.y_data)

    def generate_equidistant_test_data(self, n_dim_x, total_samples):
        x_gen = None
        for i in range(n_dim_x):
            current_x = self.x_data[:, i]
            low, high = current_x.min(), current_x.max()
            size = total_samples//10
            x_n = np.linspace(low, high, size).reshape(size, 1)
            if x_gen is None:
                x_gen = x_n
            else:
                x_gen = np.concatenate((x_gen, x_n), axis=1)

        return x_gen

    def plot_posterior(self):
        number_of_plots = self.x_data.shape[1]
        total_samples = self.x_data.shape[0]
        x_test = self.generate_equidistant_test_data(number_of_plots, total_samples)
        y_pred, sigma = self.predict(x_test)
        for i in range(number_of_plots):
            self.__plot_gp(mu=y_pred, sigma=sigma, x_data=self.x_data[:, i], y_data=self.y_data,
                           num_x_samples=total_samples//10)

    @staticmethod
    def __plot_gp(mu=None, sigma=None, x_data=None, y_data=None, x_test=None, y_test=None,
                ax=None, xlabel='$x$', ylabel='$y$',
                num_x_samples=30):
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
        if (mu is not None) and (sigma is not None):
            ax.scatter(x_test[idx], mu[idx], s=25, c=colors[0], alpha=0.9, zorder=2)
            ax.plot(x_test, mu, c=colors[0], alpha=0.9, linestyle='dashed')
            ax.fill(np.concatenate([x_test, x_test[::-1]]),
                   np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
                  alpha=.3, fc=colors[0], ec='None', label='95% confidence interval')

        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title('Wave Function', fontsize=13)
        ax.set_xlim(min(x_data) - 1, max(x_data) + 1)
        if mu is not None:
            ax.set_ylim(min(mu) - 1, max(mu) + 1)
        if y_test is not None:
            ax.set_ylim(min(y_test) - 1, max(y_test) + 1)
        plt.show()

