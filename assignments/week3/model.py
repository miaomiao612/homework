"""
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-02-07 02:51:45
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-02-07 04:13:20
FilePath: \week3\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from typing import Callable
import torch


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

    def forward(self, x):
        for i in range(self.hidden_count):
            x = self.activation(self.hidden_layers[i](x))
        x = self.output_layer(x)
        return x
