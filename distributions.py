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
        marginal_samples = self.X.sample(sample_shape)[:, 0].unsqueeze(1)
        for i in np.arange(1, self.dim):
            new_marginal_sample = self.X.sample(sample_shape)[:, i].unsqueeze_(1)
            marginal_samples = torch.cat([marginal_samples, new_marginal_sample], dim=1)

        overall_sample = torch.cat([common_sample, marginal_samples], dim=1)
        return overall_sample


class MultivariateDependentNormalThetaHalf(dist.Distribution):
    arg_constraints = {}

    def __init__(self, loc, covariance_matrix):
        super(MultivariateDependentNormalThetaHalf, self).__init__()
        self.loc = torch.tensor(ast.literal_eval(loc))
        self.covariance_matrix = torch.tensor(ast.literal_eval(covariance_matrix))
        self.dim = len(self.loc)
        self.X = dist.MultivariateNormal(self.loc, self.covariance_matrix)

    def sample(self, sample_shape):
        # Sample from the multivariate distribution
        common_sample = self.X.sample(sample_shape)
        # Sample from each marginal distribution
        marginal_samples = 1/2*self.X.sample(sample_shape)[:, 0].unsqueeze(1)+1/2*common_sample[:, 0].unsqueeze(1)
        for i in np.arange(1, self.dim):
            new_marginal_sample = 1/2*self.X.sample(sample_shape)[:, i].unsqueeze_(1)+1/2*common_sample[:, i].unsqueeze(1)
            marginal_samples = torch.cat([marginal_samples, new_marginal_sample], dim=1)

        overall_sample = torch.cat([common_sample, marginal_samples], dim=1)
        return overall_sample
