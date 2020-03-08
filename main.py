import matplotlib.pyplot as plt
from torch.optim import Adam
import torch
import numpy as np
import pickle
from gp import GP
import bec
from plots import plot_gp
from gaussian_processes_util import plot_gp_2D
from input_provider import InputProvider


def sub_plot_multiple_gp(gp, sampling_func, sample_intervals, input_variables, output_variables):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))
    ax = ax.reshape(-1)
    gs = sample_intervals

    for i in range(len(gs)):
        df = bec.get_closest_sim(sampling_func, g=gs[i])
        inp = df[input_variables]
        y_pred, sigma = gp.predict(torch.FloatTensor(inp.to_numpy()))
        # assuming the first dimension to be fixed in each iteration, and plotting the second dimension

        ax[i].scatter(df[input_variables[1]], y_pred.detach().numpy(), alpha=0.5, s=20, c='red')
        ax[i].plot(df[input_variables[1]], df[output_variables], alpha=0.6, c='green')
        ax[i].fill(np.concatenate([df[input_variables[1]], df[input_variables[1]][::-1]]),
                   np.concatenate([y_pred.detach().numpy() - 1.9600 * sigma.detach().numpy(),
                                   (y_pred.detach().numpy() + 1.9600 * sigma.detach().numpy())[::-1]]),
                   alpha=.3, fc='#ED358E', ec='None', label='95% confidence interval')
        # ax.set_title('$g$ = {:.2f}'.format(df.g[0]))
    plt.show()


def try_1D():

    ip = InputProvider()
    x_data, y_data, x_test = ip.get_1d_regression_data()

    gp = GP()
    gp.fit(x_data, y_data, True)
    gp.plot(x_test)


def try_bec():

    # get train and test data
    in_provider = InputProvider()
    harmonic_sims, tr, te, va = in_provider.get_bec_data()
    data = bec.get_within_range(tr, g_low=30, g_high=50, n=100)
    # data_test = te.sample(100)

    # get gp model and fit
    gp = GP()
    gp.fit(torch.FloatTensor(data[['g', 'x']].to_numpy()), torch.FloatTensor(data.psi.to_numpy()), True)

    # predict around one some fixed Dim
    df = bec.get_closest_sim(harmonic_sims, g=30.)
    test_gx = np.stack([30. * np.ones(df.x.shape[0]), df.x]).transpose()

    y_pred, sigma = gp.predict(torch.FloatTensor(test_gx))
    print(y_pred, sigma)

    # plot subplots with multiple fixed dimensions

    # specify input dimensions,
    # the first entry is fixed for each iteration
    # the second entry is plotted against the fixed dimension
    input_dimensions = ['g', 'x']
    out_dimensions = 'psi'
    sub_plot_multiple_gp(gp, harmonic_sims, [5, 30, 60, 90], input_dimensions, out_dimensions)

if __name__ == '__main__':
    try_bec()
