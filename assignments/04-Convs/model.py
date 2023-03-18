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
        self.conv1 = torch.nn.Conv2d(num_channels, 8, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(16 * 8 * 8, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 8 * 8)
        x = torch.nn.functional.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x
