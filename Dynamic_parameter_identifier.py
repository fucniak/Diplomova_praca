import torch
import torch.nn as nn


class Dynamic_parameter_identifier(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Dynamic_parameter_identifier, self).__init__()

        # Initialize the list of linear layers
        self.layers = nn.ModuleList()

        # Create the first layer from the input to the first hidden layer
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))

        # Dynamically add hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        # Add the final layer to the output
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)

    def forward(self, x):
        # Linearne vrstvy
        for layer in self.layers:
            x = layer(x)

        # Vystupna vrstva
        x = self.output_layer(x)
        return x
