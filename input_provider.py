import bec
import configparser
import constants
import os.path
from os import path
import pickle
import numpy as np
import torch


class InputProvider:

    def __init__(self):
        self.configs = self.load_config()


    @staticmethod
    def load_config():
        config_parser = configparser.RawConfigParser()
        config_file_path = constants.CONFIGS
        config_parser.read(config_file_path)

        bec_harmonics_path = config_parser.get(constants.CONFIG, constants.BEC_HARMONICS_PATH)

        configs = {
            constants.BEC_HARMONICS_PATH: bec_harmonics_path,
        }
        return configs

    def get_bec_data(self):
        file_path = self.configs[constants.BEC_HARMONICS_PATH]
        if path.exists(file_path):
            file = open(self.configs[constants.BEC_HARMONICS_PATH], 'rb')
            harmonic_sims = pickle.load(file)
            tr, te, va = bec.make_dataset(harmonic_sims)

        else:
            harmonic_sims = [bec.sim(g, bec.config) for g in np.linspace(0.1, 100, 300)]
            tr, te, va = bec.make_dataset(harmonic_sims)
            file_harmonics = open(file_path, 'wb')
            pickle.dump(harmonic_sims, file_harmonics)

        return harmonic_sims, tr, te, va

    def get_1d_regression_data(self):
        f = lambda x: (0.4 * x + torch.cos(2 * x) + torch.sin(x)).view(-1)
        n1, n2, ny = 6, 100, 5
        domain = (-5, 5)
        x_data = torch.distributions.Uniform(
            domain[0] + 2, domain[1] - 2).sample((n1, 1))
        y_data = f(x_data)
        x_test = torch.linspace(
            domain[0], domain[1], n2).view(-1, 1)
        y_test = f(x_test)
        return x_data, y_data, x_test, y_test

    # sin a*x
    def get_2d_sin(self):
        f = lambda a, x: (np.sin(a * x))
        x = np.linspace(-5, 5, 100)
        a = np.linspace(-3, 3, 100)
        x_data = np.stack([x,a]).transpose()
        y_data = f(a,x)
        rng = np.random.RandomState(4)
        xt = rng.uniform(-5, 5, 10)
        at = rng.uniform(-3, 3, 10)
        x_test = np.stack([xt, at]).transpose()

        return x_data, y_data, x_test

    class HigherOrderPoly:

        def get_higher_order_poly(self):
            f = self.hopoly(4)
            x_data = np.linspace(-50, 50, 200)
            y_data = f(x_data)

        def hopoly(n):
            coeff = np.random.randint(-5, 5, (n,))

            def _hopoly_fn(x):
                _sum = 0
                for i in range(n):
                    _sum += coeff[-i - 1] * (x ** (i))
                return _sum  # / _sum.max()

            return _hopoly_fn