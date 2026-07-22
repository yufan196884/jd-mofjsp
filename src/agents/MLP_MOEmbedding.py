import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super(MLPNetwork, self).__init__()
        
        # Get the layer sizes from config
        layer_sizes = config.get('Network_parameters', {}).get('layers', None)
        num_objectives = config.get('Objectives', {}).get('num_objectives', 2)

        # Build layers dynamically
        layers = []
        prev_size = input_dim
        
        # Create hidden layers
        for size in layer_sizes:
            out_dim = size-num_objectives  # Reduce output dimension by num_objectives for preference embedding
            layers.append(nn.Linear(prev_size, out_dim))
            prev_size = out_dim + num_objectives  # Add num_objectives back for the next layer to include preferences
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        # Convert list of layers to ModuleList
        self.layers = nn.ModuleList(layers)

        # Count and print the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pref = input[:, -2:]  # Extract the last 2 features as preferences
        x = input#[:, :-2]  # The rest are the main input features
        
        # Apply ReLU to all but the last layer
        for i in range(len(self.layers) - 1):
            if i == 0:
                x = self.layers[i](x)  # First layer takes the full input
            else:
                x = self.layers[i](torch.cat((x, pref), dim=-1))  # Concatenate preferences for subsequent layers
            x = F.relu(x)
        
        # Last layer has no activation (linear output)
        x = self.layers[-1](torch.cat((x, pref), dim=-1))  # Concatenate preferences for the last layer as well
        return x
