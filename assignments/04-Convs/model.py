"""
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-03-14 16:47:47
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-03-18 14:07:11
FilePath: \04-Convs\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch


class Model(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """_summary_

        Args:
            num_channels (int): _description_
            num_classes (int): _description_
        """
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 512)
        self.relu4 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
