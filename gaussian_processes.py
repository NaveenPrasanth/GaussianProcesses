import torch
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from plots import posterior_plot


class GP:
    k_variance = Variable(torch.FloatTensor([1.]), requires_grad=True)
    k_lengthscale = Variable(torch.FloatTensor([1.]), requires_grad=True)

    def __init__(self, kernel=None):
        self.x_data = None
        self.y_data = None
        self.kernel = kernel

    def _kernel(self, a, b):
        return (self.k_variance ** 2)*torch.exp(-0.5 * (a - b.T)**2 / (self.k_lengthscale**2))

    def _estimate_posterior(self, x_test):
        """Estimate Posterior (mu_2_1, cov_2_1) of GP conditioned on data (X1, y1)

        (X1, y1) : Training data points
        X2       : Test Points
        k        : kernel function
        """
        cov11 = self._kernel(self.x_data, self.x_data)
        cov12 = self._kernel(self.x_data, x_test)
        cov22 = self._kernel(x_test, x_test)
        term1, _ = torch.solve(cov12, cov11)
        mu_2_1 = torch.mm(term1.T, self.y_data.view(-1, 1)).view(-1)
        cov_2_1 = cov22 - torch.mm(term1.T, cov12)
        return mu_2_1, cov_2_1

    def _nll_loss(self):
        cov = self._kernel(self.x_data, self.x_data)
        term_1 = 0.5 * torch.log(torch.det(cov))
        term_2 = torch.mm(self.y_data.view(1, -1), torch.inverse(cov))
        term_2 = torch.mm(term_2, self.y_data.view(-1, 1)).view(1, )
        term_3 = 0.5 * self.x_data.size(0) * torch.log(2 * torch.FloatTensor([np.pi]))
        return term_1 + term_2 + term_3

    def _fit_kernel(self, epoch=10000):
        optimizer = Adam([self.k_variance, self.k_lengthscale])

        for i in range(epoch):
            optimizer.zero_grad()
            loss = self._nll_loss()
            if i % 100 == 0:
                print(i, loss.item(), 'k', self.k_variance.item(), 'l', self.k_lengthscale.item())
            loss.backward()
            optimizer.step()

    def fit(self, x_train, y_train, is_fit_needed=True):
        self.x_data = x_train
        self.y_data = y_train
        if is_fit_needed:
            self._fit_kernel()

    def predict(self, x_test):
        mu, cov = self._estimate_posterior(x_test)
        sigma = torch.sqrt(torch.diag(cov))
        return mu, sigma

    def sample(self, x_test, number_of_samples=1):
        mu, cov = self._estimate_posterior(x_test)
        y_pred = np.random.multivariate_normal(mean=mu.detach(), cov=cov.detach(), size=number_of_samples)
        return y_pred

    def plot(self):
        low, high = self.x_data.min(), self.x_data.max()
        low = low - (0.1*(high-low))
        high = high + (0.1*(high-low))

        X = torch.linspace(low, high, 100).view(-1, 1)
        mu, cov = self._estimate_posterior(X)
        sigma = torch.sqrt(torch.diag(cov))
        posterior_plot(self.x_data.detach().numpy(), self.y_data.detach().numpy(), X.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy())

