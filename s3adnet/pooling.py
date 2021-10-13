from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn


class Pooling(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass


class AbsMaxPooling(Pooling):

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        input = torch.gather(input, self.dim, torch.argmax(
            torch.abs(input), self.dim, True))
        return input.squeeze(-1)


class RobustAvgPooling(Pooling):

    def __init__(self, head=1, tail=1, dim=-1) -> None:
        super().__init__()
        self.dim = dim
        self.head = head
        self.tail = tail

    def forward(self, input: Tensor) -> Tensor:
        size = input.size(self.dim)
        upper = size - self.tail
        num = upper - self.head
        ranked = input.argsort(self.dim).argsort(self.dim)

        mask = (self.head < ranked) & (ranked <= upper)
        mask = mask.type_as(input)
        # input = input * mask
        # input = torch.sum(input, self.dim) / num
        input = input * mask * size / num
        input = torch.mean(input, self.dim)

        mask = ranked = None
        return input


def set_pooling(pool: Union[Pooling, str]) -> Pooling:
    pooling = None

    if isinstance(pool, Pooling):
        pooling = pool
    elif pool == 'absmax':
        pooling = AbsMaxPooling()
    elif pool == 'robustavg':
        pooling = RobustAvgPooling()

    else:
        raise ValueError(
            "`pool` must be an instance of Pooling or one of"
            " 'robustavg', 'absmax'!")

    return pooling
