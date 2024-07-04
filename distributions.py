import torch.distributions as dist
import torch
import ast
import numpy as np

class TwoComonotoneStandardUniforms(dist.Distribution):
    def __init__(self):
        super(TwoComonotoneStandardUniforms, self).__init__()
        self.X_1 = dist.Uniform(0, 1)
        self.X_2 = dist.Uniform(0, 1)

    # override arg_constraints
    @property
    def arg_constraints(self):
        return {}

    def sample(self, sample_shape):
        common_sample = self.X_1.sample(sample_shape)
        common_sample = torch.stack([common_sample, common_sample], dim=1)
        marginal_sample = torch.stack([self.X_1.sample(sample_shape), self.X_2.sample(sample_shape)], dim=1)
        overall_sample = torch.cat([common_sample, marginal_sample], dim=1)
        return overall_sample


class TwoIndependentStandardUniforms(dist.Distribution):

    def __init__(self):
        super(TwoIndependentStandardUniforms, self).__init__()
        self.X_1 = dist.Uniform(0, 1)
        self.X_2 = dist.Uniform(0, 1)

    @property
    def arg_constraints(self):
        return {}
    def sample(self, sample_shape):
        common_sample = torch.stack([self.X_1.sample(sample_shape), self.X_2.sample(sample_shape)], dim=1)
        marginal_sample = torch.stack([self.X_1.sample(sample_shape), self.X_2.sample(sample_shape)], dim=1)
        overall_sample = torch.cat([common_sample, marginal_sample], dim=1)
        return overall_sample


class MultivariateDependentNormal(dist.Distribution):
    arg_constraints = {}
    def __init__(self, loc, covariance_matrix):
        super(MultivariateDependentNormal, self).__init__()
        self.loc = torch.tensor(ast.literal_eval(loc))
        self.covariance_matrix = torch.tensor(ast.literal_eval(covariance_matrix))
        self.dim = len(self.loc)
        self.X = dist.MultivariateNormal(self.loc, self.covariance_matrix)

    def sample(self, sample_shape):
        # Sample from the multivariate distribution
        common_sample = self.X.sample(sample_shape)
        # Sample from each marginal distribution
        marginal_dist = dist.Normal(self.X.mean[0], torch.sqrt(self.X.covariance_matrix[0, 0]))
        marginal_samples = marginal_dist.sample(sample_shape)
        for i in np.arange(1, self.dim):
            marginal_dist = dist.Normal(self.X.mean[i], torch.sqrt(self.X.covariance_matrix[i, i]))
            marginal_samples = torch.stack([marginal_samples, marginal_dist.sample(sample_shape)], dim=1)
        overall_sample = torch.cat([common_sample, marginal_samples], dim=1)
        return overall_sample
