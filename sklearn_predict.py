from examples.regression_1d import *
from examples.higher_order_poly  import *
from examples.bec_data import *
from examples.sin_2d import *
from torch_utils import *
import bec
import torch
from plots import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp import GP


class GPSelect:
    def __init__(self, components=1, name="sklearn"):
        if name == "sklearn":
            kernel = C(1.0, (1e-3, 1e3)) * RBF([5], (1e-2, 1e2))
            if components == 2:
                kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5, 5, 5], (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

        elif name == 'torchgp':
            self.gp = GP()

    def get_gp(self):
        return self.gp


def sub_plot_multiple_gp(gp, sampling_func, sample_intervals, input_variables, output_variables):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))
    ax = ax.reshape(-1)
    gs = sample_intervals

    for i in range(len(gs)):
        df = bec.get_closest_sim(sampling_func, g=gs[i])
        inp = df[input_variables]
        y_pred, sigma = gp.predict(torch.FloatTensor(inp.to_numpy()), return_std=True)
        sigma[sigma < 0.] = 0.
        # assuming the first dimension to be fixed in each iteration, and plotting the second dimension

        ax[i].scatter(df[input_variables[1]], y_pred, alpha=0.5, s=20, c='red')
        ax[i].plot(df[input_variables[1]], df[output_variables], alpha=0.6, c='green')
        ax[i].fill(np.concatenate([df[input_variables[1]], df[input_variables[1]][::-1]]),
                   np.concatenate([y_pred - 1.9600 * sigma,
                                   (y_pred + 1.9600 * sigma)[::-1]]),
                   alpha=.3, fc='#ED358E', ec='None', label='95% confidence interval')
        # ax.set_title('$g$ = {:.2f}'.format(df.g[0]))
    plt.show()


def try_1d(gp = None):

    x_data, y_data, x_test, y_test = get_1d_regression_data()
    gp.fit(x_data, y_data)

    y_pred, sigma = gp.predict(x_test, return_std=True)
    sigma[sigma < 0.] = 0.
    plot_gp(y_pred, sigma, x_data.numpy(), y_data.numpy(), x_test.numpy(), y_test.numpy())


def try_bec(gp=None):

    # get train and test data
    harmonic_sims, tr, te, va = get_bec_data()
    data = bec.get_within_range(tr, g_low=30, g_high=90, n=500)
    # data_test = te.sample(100)

    # get gp model and fit
    gp.fit(torch.FloatTensor(data[['g', 'x']].to_numpy()), torch.FloatTensor(data.psi.to_numpy()))

    # predict around one some fixed Dim
    df = bec.get_closest_sim(harmonic_sims, g=30.)
    test_gx = np.stack([30. * np.ones(df.x.shape[0]), df.x]).transpose()

    y_pred, sigma = gp.predict(torch.FloatTensor(test_gx), return_std=True)
    sigma[sigma < 0.] = 0.
    print(y_pred, sigma)

    # plot subplots with multiple fixed dimensions

    # specify input dimensions,
    # the first entry is fixed for each iteration
    # the second entry is plotted against the fixed dimension
    input_dimensions = ['g', 'x']
    out_dimensions = 'psi'
    sub_plot_multiple_gp(gp, harmonic_sims, np.linspace(30,90,4), input_dimensions, out_dimensions)


def try_2d_sin(gp = None):

    x_data, y_data, x_test, y_test = get_2d_sin()
    x_data = t(x_data)
    y_data = t(y_data)
    x_test = t(x_test)
    y_test = t(y_test)

    gp.fit(x_data, y_data)
    y_pred, sigma = gp.predict(x_test, return_std=True)
    sigma[sigma < 0.] = 0.
    plot_gp(y_pred, sigma, x_data, y_data, x_test, y_test)


def try_hopoly_1d(gp = None):
    x_data, y_data, x_test, y_test = get_higher_order_poly()
    x_data = t(x_data).view(-1,1)
    y_data = t(y_data).view(-1,1)
    x_test = t(x_test).view(-1,1)
    y_test = t(y_test).view(-1,1)
    gp.fit(x_data, y_data)
    y_pred, sigma = gp.predict(x_test, return_std=True)
    sigma[sigma < 0.] = 0.
    plot_gp(y_pred, sigma, x_data.numpy(), y_data.numpy(), x_test.numpy(), y_test.numpy(), num_x_samples=10)


if __name__ == '__main__':
    gp = GPSelect("sklearn")
    gp = gp.get_gp()
    #try_1d(gp=gp)
    #try_bec(gp=gp)
    try_bec(gp=gp)