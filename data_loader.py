import torch
import torch.distributions as dist
def data_generator_from_distribution(batch_size, distribution: dist.Distribution):
    """
    Generates data from a given distribution.

    :param batch_size: The size of each batch to generate.
    :param distribution: A torch.distributions.Distribution object representing the desired distribution.
    :return: A generator that yields batches of samples from the specified distribution.
    """
    while True:
        yield distribution.sample((batch_size,))

# mixture_dist = dist.MixtureSameFamily(
#     dist.Categorical(torch.ones(2,)),  # Uniform mixture weight
#     dist.Independent(dist.Normal(torch.tensor([0.0, 5.0]), torch.tensor([1.0, 2.0])), 1)
# )
# data_gen = data_generator_from_distribution(32, mixture_dist)
# data = next(data_gen)
# print(data)