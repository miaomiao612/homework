"""
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-02-06 13:51:45
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-02-07 13:24:23
FilePath: \week3\model_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
"""
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-02-06 13:51:45
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-02-07 12:56:22
FilePath: \week3\model_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.

    """
    return MLP(
        input_dim,
        200,
        output_dim,
        1,
        torch.nn.LeakyReLU,
        torch.nn.init.kaiming_uniform_,
    )
