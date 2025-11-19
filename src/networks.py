import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes = (128, 128)):
        """
        A simple feedforward neural network that maps states -> Q-values.

        input_dim: size of the state vector (e.g. 4 for CartPole)
        output_dim: number of actions (e.g. 2 for CartPole)
        hidden_sizes: sizes of hidden layers, e.g. (128, 128) means:
        input -> 128 -> 128 -> output
        """
        
        super().__init__()
        
        layers = []
        last_dim = input_dim
        
        #Build hidden layers
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        
        #Output layer for Q-values
        layers.append(nn.Linear(last_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

