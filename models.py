import torch
import torch.nn as nn

class RiskAggregationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, output_dim=1):
        super(RiskAggregationNN, self).__init__()
        hidden_dim = hidden_dim

        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(input_dim, hidden_dim))
        self.model.add_module('relu0', nn.ReLU())

        for i in range(num_hidden_layers):
            self.model.add_module(f'hidden{i + 1}', nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module(f'relu{i + 1}', nn.ReLU())

        self.model.add_module('output', nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.model(x)

class ParallelRiskAggregationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(ParallelRiskAggregationNN, self).__init__()
        self.input_dim = input_dim
        self.models_h = nn.ModuleList([RiskAggregationNN(1, hidden_dim, num_hidden_layers, ) for _ in range(input_dim)])
        self.model_g = RiskAggregationNN(input_dim, hidden_dim, num_hidden_layers)

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
