import torch
from typing import Dict
import random
import os
import numpy as np



def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TensorboardAggregator:
    """
        Log average value periodically instead of logging on every batch
    """
    def __init__(self, writer, every=1):
        self.writer = writer
        self.every = every
        self.step = 0
        self.scalars = None

    def log(self, scalars: Dict[str, float]):
        self.step += 1
        if self.scalars is None:
            self.scalars = scalars.copy()
        else:
            for k, v in scalars.items():
                self.scalars[k] += v
        if self.step % self.every == 0:
            for k, v in self.scalars.items():
                self.writer.add_scalar(k, v / self.every, self.step)
            self.scalars = None


