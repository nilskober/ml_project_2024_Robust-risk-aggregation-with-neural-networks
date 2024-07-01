import torch
import torch.distributions as dist

from data_loader import data_generator_from_distribution
from loss_functions import loss_function_empirical_integral
from models import ParallelRiskAggregationNN
from optimization_pipeline import optimize_model


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

if __name__ == '__main__':
    model = ParallelRiskAggregationNN(2)
    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device("cpu")
    print(device)
    model.to(device)
    # define f as maximum of two values (for tensors)
    f = lambda x: torch.max(x[:, 0], x[:, 1])
    distribution__example_4_1 = TwoComonotoneStandardUniforms()
    data_gen = data_generator_from_distribution(128, distribution__example_4_1)
    optimize_model(device, model, loss_function_empirical_integral, data_gen, {'rho': 0.5, 'gamma': 1280, 'f': f, 'input_dim': 2}, 20000, 15000, 0.0001, print_every=100)
    # Save model to disk
    torch.save(model.state_dict(), 'model_example_4_1.pth')
    print("Test model")
    model.eval()
    data_gen_test = data_generator_from_distribution(2**16, distribution__example_4_1)
    data_test = next(data_gen_test)
    inputs = data_test.to(device)
    outputs = model(inputs)
    loss = loss_function_empirical_integral(inputs, outputs, 0.5, 1280, f, 2)
    print(f'Loss on test data: {loss.item():.4f}')


