import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, pref_dim=2, last_block=False):
        super(FiLMBlock, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)

        # FiLM parameters
        self.film_params = nn.Linear(pref_dim, 2 * hidden_dim)

        self.last_block = last_block
        if not last_block:
            self.relu = nn.ReLU()
        
    def forward(self, x, pref):

        # Compute FiLM parameters
        x = self.linear(x)

        film_params = self.film_params(pref)
        gamma, beta = torch.split(film_params, x.size(-1), dim=-1)
        x = gamma * x + beta
        if not self.last_block:
            x = self.relu(x)
        return x

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: dict, embedd_pref=True):
        super(MLPNetwork, self).__init__()
        
        # Get the layer sizes from config
        layer_sizes = config.get('Network_parameters', {}).get('layers', None)
        
        self.embedd_pref = embedd_pref

        # Build layers dynamically
        layers = []
        prev_size = input_dim
        
        # Create hidden layers
        for size in layer_sizes:
            if embedd_pref:
                out_dim = size - 2  # Reduce output dimension by 2 for preference embedding
            else:
                out_dim = size
            layers.append(FiLMBlock(prev_size, out_dim))
            prev_size = size  # Add 2 back for the next layer to include preferences
        
        # Add output layer
        layers.append(FiLMBlock(prev_size, output_dim, last_block=True))
        
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
            if i!= 0 and self.embedd_pref:
                x = torch.cat((x, pref), dim=-1)  # Concatenate preferences for subsequent layers
            x = self.layers[i](x, pref)
        
        # Last layer has no activation (linear output)
        if self.embedd_pref:
            x = torch.cat((x, pref), dim=-1)  # Concatenate preferences for the last layer as well
        x = self.layers[-1](x, pref)
        return x
