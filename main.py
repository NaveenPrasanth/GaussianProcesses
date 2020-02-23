import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam
import numpy as np

from gaussian_processes import GP

gp = GP()
f_sin = lambda x: torch.sin(x).view(-1)
n1, n2, ny = 8, 75, 5
domain = (-6, 6)
X_data = Variable(
    torch.distributions.Uniform(
        domain[0] + 2, domain[1] - 2).sample((n1, 1)).requires_grad_(True)
)
y_data = Variable(
    f_sin(X_data).requires_grad_(True)
)
X_test = Variable(
    torch.linspace(
        domain[0], domain[1], n2).view(-1, 1).requires_grad_(True)
)

gp.fit(X_data, y_data, False)
gp.plot()
print(gp.predict(torch.FloatTensor([10.]).view(-1, 1)))

gp.fit(X_data, y_data)
gp.plot()

