import torch
import torch.nn as nn

class RiskAggregationNN(nn.Module):
    def __init__(self, input_dim, output_dim=1):
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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ParallelRiskAggregationNN(nn.Module):
    def __init__(self, input_dim, lambda_init=1.0):
        super(ParallelRiskAggregationNN, self).__init__()
        self.input_dim = input_dim
        self.models_h = nn.ModuleList([RiskAggregationNN(1) for _ in range(input_dim)])
        self.model_g = RiskAggregationNN(input_dim)

    def forward(self, x):
        # Compute the output of each model in parallel for the i-th dimension of the input
        # TODO: Make this more efficient using torch operations
        first_half = x[:, :self.input_dim]
        second_half = x[:, self.input_dim:]
        outputs_h = [model(second_half[:, i].unsqueeze(1)) for i, model in enumerate(self.models_h)]
        outputs_g = self.model_g(first_half)
        outputs_h_concat = torch.cat(outputs_h, dim=1)
        outputs = {
            'h': outputs_h_concat,
            'g': outputs_g
        }
        return outputs

