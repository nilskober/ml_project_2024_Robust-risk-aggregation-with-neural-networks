import torch.distributions as dist
import torch


class TwoComonotoneStandardUniforms(dist.Distribution):
    def __init__(self):
        super(TwoComonotoneStandardUniforms, self).__init__()
        self.X_1 = dist.Uniform(0, 1)
        self.X_2 = dist.Uniform(0, 1)

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


    def sample(self, sample_shape):
        common_sample = torch.stack([self.X_1.sample(sample_shape), self.X_2.sample(sample_shape)], dim=1)
        marginal_sample = torch.stack([self.X_1.sample(sample_shape), self.X_2.sample(sample_shape)], dim=1)
        overall_sample = torch.cat([common_sample, marginal_sample], dim=1)
        return overall_sample

