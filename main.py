import torch.optim as optim

# Set the device to GPU if available, otherwise to CPU
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


# Example usage
input_dim = 10  # Change this to the actual input dimension
model = RiskAggregationNN(input_dim).to(device)