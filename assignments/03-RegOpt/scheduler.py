from typing import List
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """_summary_

    Args:
        _LRScheduler (_type_): _description_
    """

    def __init__(
        self, optimizer, step_size=782 * 5, gamma=0.6, last_epoch=-1, verbose=False
    ):
        """_summary_

        Args:
            optimizer (_type_): _description_
            step_size (_type_, optional): _description_. Defaults to 783*4.
            gamma (float, optional): _description_. Defaults to 0.7.
            last_epoch (int, optional): _description_. Defaults to -1.
            verbose (bool, optional): _description_. Defaults to False.
        """

        self.step_size = step_size
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """_summary_

        Returns:
            _type_: _description_
        """

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
