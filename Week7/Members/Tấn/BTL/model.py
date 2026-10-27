import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_width=64, hidden_layers=2, activation="tanh"):
        super().__init__()
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(activations[activation.lower()]())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
