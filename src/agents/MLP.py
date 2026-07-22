import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super(MLPNetwork, self).__init__()
        
        # Get the layer sizes from config
        layer_sizes = config.get('Network_parameters', {}).get('layers', None)
        
        # Build layers dynamically
        layers = []
        prev_size = input_dim
        
        # Create hidden layers
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        # Convert list of layers to ModuleList
        self.layers = nn.ModuleList(layers)

        # Count and print the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        
        # Apply ReLU to all but the last layer
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Last layer has no activation (linear output)
        x = self.layers[-1](x)
        return x
