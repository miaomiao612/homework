from typing import Callable
import torch
from torch import nn


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.hidden_count = hidden_count
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(hidden_count):
            self.hidden_layers.append(torch.nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        self.output_layer = torch.nn.Linear(hidden_size, num_classes)
        self.activation = activation()
        torch.nn.init.ones_(self.output_layer.weight)
        torch.nn.init.ones_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.hidden_count):
            x = self.activation(self.hidden_layers[i](x))
        x = self.output_layer(x)
        return x
