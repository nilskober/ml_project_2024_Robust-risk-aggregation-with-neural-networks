import numpy as np
import torch


# class IndependentMultivariateDistribution(torch.distributions.Distribution):
#     def __init__(self, *distributions):
#         self.distributions = distributions
#
#     def sample(self, sample_shape)
#         return torch.stack([d.sample(n) for d in self.distributions])
#
#     def logpdf(self, X):
#         return sum(d.logpdf(X[:, i]) for i, d in enumerate(self.distributions))