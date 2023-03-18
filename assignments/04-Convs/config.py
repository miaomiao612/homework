"""
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-03-14 16:47:47
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-03-18 14:04:06
FilePath: \04-Convs\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 128
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

    transforms = Compose([ToTensor()])
