import torch
import torch.nn as nn

class RiskAggregationNN(nn.Module):
    def __init__(self, input_dim):
        super(RiskAggregationNN, self).__init__()
        hidden_dim = 64 * input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

