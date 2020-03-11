import torch
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import kernel

from plots import plot_gp, collate_plots
from utils import *

from tqdm import tqdm


class GP:
    k_variance = Variable(torch.FloatTensor([0.3218]), requires_grad=True)
    k_lengthscale = Variable(torch.FloatTensor([2.4857]), requires_grad=True)

    def __init__(self, kernel=None):
        self.x_data = None
        self.y_data = None
        self.kernel = kernel

    def _kernel(self, a, b):
        # return (self.k_variance ** 2)*torch.exp(-0.5 * (a - b.T)**2 / (self.k_lengthscale**2))
        return kernel.rbf(a, b,
            variance=self.k_variance, lengthscale=self.k_lengthscale)

    def _estimate_posterior(self, x_test):
        """Estimate Posterior (mu, cov) of GP conditioned on data

        x_test : test points
        """
        cov11 = self._kernel(self.x_data, self.x_data)
        cov12 = self._kernel(self.x_data, x_test)
        cov22 = self._kernel(x_test, x_test)
        term1, _ = torch.solve(cov12, cov11)
        mu_2_1 = torch.mm(term1.T, self.y_data.view(-1, 1)).view(-1)
        cov_2_1 = cov22 - torch.mm(term1.T, cov12)
        return mu_2_1, cov_2_1

    def _estimate_posterior_stable(self, x_test):
        """Estimate Posterior (mu, cov) of GP conditioned on data

        x_test : test points
        """
        cov11 = self._kernel(self.x_data, self.x_data)
        cov12 = self._kernel(self.x_data, x_test)
        cov22 = self._kernel(x_test, x_test)
        # cholesky decomposition of cov11
        L = cholesky(cov11)
        A = torch.triangular_solve(cov12, L, upper=False)[0]
        V = torch.triangular_solve(self.y_data.view(-1, 1), L, upper=False)[0]
        # sufficient statistics
        mu = A.t() @ V
        cov = cov22 - A.t() @ A
        return mu.view(-1), cov

    def _nll_stable(self):

        cov = self._kernel(self.x_data, self.x_data)
        L = cholesky(cov)
        assert len(L.shape) == 2, L.shape
        term_1 = torch.log(torch.diag(L)).sum()
        term_2 = lstq(self.y_data, L)
        term_2 = lstq(term_2, L.T).view(-1, 1)
        term_2 = 0.5 * torch.mm(self.y_data.view(1, -1), term_2)
        term_3 = 0.5 * self.x_data.size(0) * torch.log(2 * torch.FloatTensor([np.pi]))
        return (term_1 + term_2 + term_3)

    def _nll_loss(self):
        cov = self._kernel(self.x_data, self.x_data)
        term_1 = 0.5 * torch.log(torch.det(cov))
        term_2 = torch.mm(self.y_data.view(1, -1), torch.inverse(cov))
        term_2 = torch.mm(term_2, self.y_data.view(-1, 1)).view(1, )
        term_3 = 0.5 * self.x_data.size(0) * torch.log(2 * torch.FloatTensor([np.pi]))
        return term_1 + term_2 + term_3

    def _fit_kernel(self, epochs=1000):
        optimizer = Adam([self.k_variance, self.k_lengthscale])

        pbar = tqdm(range(epochs))
        for i in pbar:
          optimizer.zero_grad()
          loss = self._nll_stable()
          loss.backward()
          optimizer.step()
          if i % 10 == 0:
            pbar.set_description(
                'loss : {:3.4f} | kernel params : [ {:2.4f}, {:2.4f} ]'.format(
                loss.item(), self.k_variance.item(), self.k_lengthscale.item()
                ))

    def fit(self, x_train, y_train, is_fit_needed=True, epochs=100):
        self.x_data = x_train
        self.y_data = y_train
        if is_fit_needed:
            self._fit_kernel(epochs=epochs)

    def predict(self, x_test):
        mu, cov = self._estimate_posterior_stable(x_test)
        diag = torch.diag(cov)

        # numerical error check
        diag_negatives = torch.where(diag < 0., diag, -1e-10 + torch.zeros_like(diag))
        # if torch.abs(diag_negatives).min() < 1e-4:
        #  print('WARNING : Negative values are of high magnitude')
        print('INFO : Highest Negative Variance', diag_negatives.min().item())

        diag_clipped = torch.where(diag > 0., diag, torch.zeros_like(diag))
        sigma = torch.sqrt(diag_clipped)
        return mu, sigma

    def sample(self, x_test, number_of_samples=1):
        mu, cov = self._estimate_posterior_stable(x_test)
        y_pred = np.random.multivariate_normal(
            mean=mu.detach(), cov=cov.detach(), size=number_of_samples)
        return y_pred

    def plot(self, X, y=None, include_cov=False):
        low, high = self.x_data.min(), self.x_data.max()
        low = low - (0.1*(high-low))
        high = high + (0.1*(high-low))

        #X = torch.linspace(low, high, 100).view(-1, 1)
        mu, cov = self._estimate_posterior_stable(X)
        diag = torch.diag(cov)

        # numerical error check
        diag_negatives = torch.where(diag < 0., diag, -1e-10 + torch.zeros_like(diag))
        # if torch.abs(diag_negatives).min() < 1e-4:
        #  print('WARNING : Negative values are of high magnitude')
        print('INFO : Highest Negative Variance', diag_negatives.min().item())

        diag_clipped = torch.where(diag > 0., diag, torch.zeros_like(diag))
        sigma = torch.sqrt(diag_clipped)
        """
        plot_gp(
            mu.detach().numpy(), sigma.detach().numpy(),
            x_data=self.x_data.detach().numpy(),
            y_data=self.y_data.detach().numpy(),
            x_test=X.detach().numpy(), y_test=y,
            num_x_samples=35
            )
        """
        collate_plots(mu=mu.detach().numpy(), x_test=X.detach().numpy(),
            cov=cov.detach().numpy(), K=self._kernel(X, X).detach().numpy(),
            x_data=self.x_data.detach().numpy(),
            y_data=self.y_data.detach().numpy())
