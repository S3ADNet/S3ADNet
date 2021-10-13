from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init

from .utils import bulk_forward
from .pooling import Pooling, set_pooling


class MCCLayer(nn.Module):

    def __init__(self, embedding_size: int,
                 num_concepts: int, bias: bool = True) -> None:
        """Multi-conceptual context layer

        Parameters
        ----------
        embedding_size : int
            :math:`M`
        num_concepts : int
            :math:`C`
        bias : bool, optional
            bias for each concept, by default True

        Returns
        -------
        Callable[[Tensor], Tensor]
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.num_concepts = num_concepts

        self.weight = nn.Parameter(Tensor(
            embedding_size, num_concepts, embedding_size))  # (M, C, M)
        if bias:
            self.bias = nn.Parameter(Tensor(
                1, 1, num_concepts, embedding_size))
        else:
            self.register_parameter('bias', None)

        self.activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self) -> None:

        bound = 1. / self.embedding_size
        init.kaiming_uniform_(self.weight, nonlinearity='tanh')

        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        r"""

        Parameters
        ----------
        x : Tensor
            :math:`(B, L, M)`

        Returns
        -------
        Tensor
            :math:`(B, L, C, L)`
        """
        x_T = x.transpose(1, 2).unsqueeze(1)        # (B, M, L)

        x = torch.tensordot(
            x, self.weight, dims=([2], [0]))  # (B, L, C, M)

        if self.bias is not None:
            x += self.bias

        x = self.activation(x)

        x = x @ x_T     # (B, L, C, L)
        x_T = None

        return x


class AnomalyGate(nn.Module):

    def __init__(self, embedding_size: int, num_concepts: int,
                 context_pool: Union[str, Pooling] = 'robustavg',
                 concept_pool: Union[str, Pooling] = 'absmax',
                 bias: bool = True) -> None:

        super().__init__()

        self.mcc = MCCLayer(embedding_size, num_concepts, bias)
        self.context_pool = set_pooling(context_pool)
        self.concept_pool = set_pooling(concept_pool)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            :math:`(B, L, M)`

        Returns
        -------
        Tensor
        """

        embedding_size = x.size(-1)
        x = self.mcc(x)                 # (B, L, C, L)

        x = self.context_pool(x)         # (B, L, C)
        x = self.concept_pool(x)         # (B, L)

        x /= math.sqrt(embedding_size)

        return x


class S3ADNet(nn.Module):

    def __init__(
            self,
            basemodel: nn.Module,
            embedding_size: int,
            num_concepts: int = 1,
            hidden_size: Optional[int] = None,
            context_pool: Union[str, Pooling] = 'robustavg',
            concept_pool: Union[str, Pooling] = 'absmax',
            bias: bool = True,
            compute_gates: bool = False,
            output_probs: bool = True,
            bulk_size: int = 256) -> None:
        """Self-supervised Sequential Anomaly Detection Network

        Parameters
        ----------
        basemodel : nn.Module
            Encodes every data points
        embedding_size : int
            the size of each embedding
        num_concepts : int, optional
            by default 1
        hidden_size : Optional[int], optional
            [description], by default None
        context_pool : Union[str, Pooling], optional
            contextual pooling, by default 'avg'
        concept_pool : Union[str, Pooling], optional
            multi-conceptual pooling, by default 'avg'
        bias : bool, optional
            Bias for the networks, by default True
        compute_gates : bool, optional
            Whether calculates anomaly results, by default False
        output_probs : bool, optional
            Whether output anomaly probablities rather than anomaly scores, by default True
        bulk_size : bool, optional
            the number of data points for parallel encoding, by default 256
        """

        super().__init__()

        self.base = basemodel
        hidden_size = hidden_size if hidden_size is not None else embedding_size
        self.compute_gates = compute_gates
        self.output_probs = output_probs
        self.bulk_size = bulk_size

        self.gate = AnomalyGate(
            embedding_size, num_concepts,
            concept_pool=concept_pool,
            context_pool=context_pool,
            bias=bias)

    def forward(self, input: Tensor) -> Union[Tensor,
                                              tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        input : Tensor
            :math:`(B, L, *)`

        Returns
        -------
        Union[Tensor, tuple[Tensor, Tensor]]
            (projection) or (projection, anomaly_value)
        """

        input = bulk_forward(self.base, input, self.bulk_size)

        if self.compute_gates:
            gates = self.gate(input)
            if self.output_probs:
                gates = torch.sigmoid(gates)
            return input, gates

        return input
