from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class CARELoss(_Loss):

    def __init__(self, seq_len: int,
                 adaptation: Union[str, Callable[[int], float]],
                 lookahead: int = 1, penalty: float = .2,
                 adaptation_coef=1., bidirectional: bool = False,
                 K: float = 1., C: float = 1.,
                 Alpha: float = math.e, Lambda: float = 1.1,
                 eps: float = 1e-8, reduction: str = "mean") -> None:
        """Context-adaptive relative entropy loss

        Parameters
        ----------
        seq_len : int
            The length of a sequence
        adaptation : Union[str, Callable[[int], float]], optional
            Adapts the strengths of relationships: ``'linear'``| ``'constant'``| ``'sqrt'``| ``'log1p'``| ``'exp'``, or a callable which receives the difference between two steps and outputs a float number
        lookahead : int, optional
            The number of steps to look ahead for the sequential relationships, by default 1
        penalty : float, optional
            Constrains the entropy maximization, by default .1
        bidirectional : bool, optional
            Whether calculates bidirectional relationships. by default False
        K : float, optional
            The coeffcient for the linear adaptation, by default 1.
        C : float, optional
            The number for the constant adaptation, by default 1.
        Alpha : float, optional
            The base for the log1p adaptation, by default math.e
        Lambda : float, optional
            The base for the exp adaptation, by default 1.1
        eps : float, optional
            The small number to avoid :math:`log(0)`, by default 1e-8
        reduction : str, optional
            Specifies the reduction to apply to the output:``'none'`` | ``'mean'`` | ``'sum'``., by default "mean"

        Raises
        ------
        ValueError
        """
        super().__init__(reduction=reduction)
        self.seq_len = seq_len
        self.penalty = penalty
        self.adaptation_coef = adaptation_coef
        self.bidirectional = bidirectional
        self.steps = min(max(lookahead, 1), seq_len - 1)

        self.eps = eps

        if adaptation == "linear":
            self.adaptation = lambda delta: K * delta
        elif adaptation == "constant":
            self.adaptation = lambda delta: C
        elif adaptation == "sqrt":
            self.adaptation = lambda delta: math.sqrt(delta)
        elif adaptation == "log1p":
            self.adaptation = lambda delta: (
                math.log1p(delta) / math.log(Alpha))
        elif adaptation == "exp":
            self.adaptation = lambda delta: Lambda ** delta
        elif isinstance(adaptation, function):
            self.adaptation = adaptation
        else:
            raise ValueError(
                "`adaptation` must be 'linear', 'constant', 'sqrt', 'log1p', 'exp', or a callable.")

        self._setup_attemperation()

    def _setup_attemperation(self) -> None:

        attemperate_per_step = sum(np.diag(np.full(
            self.seq_len - 1 - i, 1. /
            self.adaptation_coef / self.adaptation(i + 1),
            dtype=np.float32), i) for i in range(self.steps))
        step_mask = (attemperate_per_step != 0).astype(np.float32)
        step_average = step_mask / step_mask.sum(1, keepdims=True)

        step_indices = step_mask.flatten().nonzero()[0]

        step_weight = torch.from_numpy(step_average.flatten()[
            step_indices][np.newaxis])
        adaptation = torch.from_numpy(attemperate_per_step.flatten()[
            step_indices][np.newaxis])
        step_indices = torch.from_numpy(step_indices)

        self.register_buffer('_step_weight', step_weight)
        self.register_buffer('_attemperation', adaptation)
        self.register_buffer('_step_indices', step_indices)

    def _staircase(self, input: Tensor) -> Tensor:
        return torch.index_select(input, 1, self._step_indices)

    def _plogq(self, p: Tensor, q: Optional[Tensor] = None) -> Tensor:
        if q is None:
            q = p
        return p * torch.log(q.clamp(self.eps))     # avoid log(0)

    def forward_oneway(self, input: Tensor, gates: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input : Tensor
            (B, L, M)
        gates : Tensor
            (B, L)

        Returns
        -------
        Tensor
        """

        batch_size = input.size(0)
        seq_size = self.seq_len - 1
        seq_size_squared = seq_size ** 2

        # staircase projections
        z_i = input[:, :-1, None].repeat(1, 1, seq_size, 1)  # (B, L-1, L-1, M)
        z_i = z_i.reshape(batch_size, seq_size_squared, -1)
        z_i = self._staircase(z_i)

        z_j = input[:, 1:, None].repeat(1, 1, seq_size, 1)
        z_j = z_j.transpose_(1, 2)
        z_j = z_j.reshape(batch_size, seq_size_squared, -1)
        z_j = self._staircase(z_j)

        # propabilities of relationship
        sim = F.cosine_similarity(z_i, z_j, dim=-1)
        rel = sim * self._attemperation
        notQ = torch.sigmoid(rel)
        Q = 1. - notQ

        rel = sim = z_i = z_j = None

        # staircase gates of anomaly
        g_i = gates[:, :-1, None].repeat(1, 1, seq_size)
        g_i = g_i.reshape(batch_size, -1)
        g_i = self._staircase(g_i)

        g_j = gates[:, 1:, None].repeat(1, 1, seq_size)
        g_j = g_j.transpose_(1, 2)
        g_j = g_j.reshape(batch_size, -1)
        g_j = self._staircase(g_j)

        P = g_i + g_j - g_i * g_j
        notP = 1. - P

        g_i = g_j = None

        # altered biKLD
        loss = self.penalty * (self._plogq(P) + self._plogq(notP))
        loss -= self._plogq(P, Q) + self._plogq(notP, notQ)
        loss += self.penalty * (self._plogq(Q) + self._plogq(notQ))
        loss -= self._plogq(Q, P) + self._plogq(notQ, notP)
        loss *= 0.5

        # altered JSD
        # M = 0.5 * (P + Q)
        # notM = 1. - M
        # loss = self.penalty * (self._plogq(P) + self._plogq(notP))
        # loss -= self._plogq(P, M) + self._plogq(notP, notM)
        # loss += self.penalty * (self._plogq(Q) + self._plogq(notQ))
        # loss -= self._plogq(Q, M) + self._plogq(notQ, notM)
        # loss *= 0.5

        P = notP = Q = notQ = None

        loss = self._step_weight * loss

        if self.reduction == "mean":
            loss = torch.sum(loss, dim=1) / (self.seq_len - 1)
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

    def forward(self, input: Tensor, gates: Tensor) -> Tensor:

        loss = self.forward_oneway(input, gates)
        if self.bidirectional:
            loss += self.forward_oneway(
                torch.flip(input, [1]), torch.flip(gates, [1]))
            loss *= 0.5
        return loss


class SeNTXentLoss(_Loss):

    def __init__(self, temperature: float = 1.,
                 reduction: str = "mean") -> None:
        """Sequential NT-Xent loss

        Parameters
        ----------
        temperature : float, optional
            Scales the similarities, by default 1.
        reduction : str, optional
            Specifies the reduction to apply to the output:``'none'`` | ``'mean'`` | ``'sum'``., by default "mean"
        """
        super().__init__(reduction=reduction)
        self.temperature = temperature

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:

        batch_size = input1.size(0)

        joined = torch.cat([input1, input2], dim=0)  # (2B, L, M)
        joined = F.cosine_similarity(
            joined.unsqueeze(1), joined.unsqueeze(0), dim=-1)  # (2B, 2B, L)
        joined -= torch.eye(2 * batch_size, device=input1.device).unsqueeze_(2)
        joined /= self.temperature

        joined = torch.logsumexp(joined, dim=1)     # (2B, L)

        sim = F.cosine_similarity(
            input1, input2, dim=-1) / self.temperature    # (B, L)
        loss = - sim + 0.5 * (joined[:batch_size] + joined[batch_size:])

        sim = joined = None

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class SequentialKLDiv(_Loss):
    def __init__(self, bidirectional=True, max=20.,
                 reduction: str = "mean") -> None:
        """Sequential Kullback–Leibler Divergence

        Parameters
        ----------
        bidirectional : bool, optional
            Whether calculates bidirectional KL divergence, by default True
        reduction : str, optional
            Specifies the reduction to apply to the output:``'none'`` | ``'mean'`` | ``'sum'``., by default "mean"
        """
        super().__init__(reduction=reduction)
        self.bidirectional = bidirectional
        self.max = max

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        batch_size = input1.size(0)
        dist_size = input1.size(-1)
        input1 = input1.reshape(-1, dist_size)      # (B*L, M)
        input2 = input2.reshape(-1, dist_size)      # (B*L, M)

        input1 = F.log_softmax(input1, dim=-1)
        input2 = F.log_softmax(input2, dim=-1)

        kld = F.kl_div(input1, input2,
                       reduction='none', log_target=True)
        if self.bidirectional:
            kld += F.kl_div(input2, input1,
                            reduction='none', log_target=True)
            kld *= 0.5

        kld = kld.clip_(max=self.max)
        kld = torch.sum(kld, dim=1)      # (B*L,)
        kld = kld.view(batch_size, -1)   # (B, L)

        if "mean" == self.reduction:
            kld = torch.mean(kld)
        elif "sum" == self.reduction:
            kld = torch.sum(kld)

        return kld.clip_(max=self.max)


class SequentialJSDiv(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        batch_size = input1.size(0)
        dist_size = input1.size(-1)
        input1 = input1.reshape(-1, dist_size)      # (B*L, F)
        input2 = input2.reshape(-1, dist_size)      # (B*L, F)

        input1 = F.log_softmax(input1, dim=-1)
        input2 = F.log_softmax(input2, dim=-1)
        mean = 0.5 * (torch.exp(input1) + torch.exp(input2))

        jsd = F.kl_div(input1, mean, reduction='none')
        jsd += F.kl_div(input2, mean, reduction='none')
        jsd *= 0.5

        jsd = torch.sum(jsd, dim=1)      # (B*L,)
        jsd = jsd.view(batch_size, -1)   # (B, L)

        if "mean" == self.reduction:
            jsd = torch.mean(jsd)
        elif "sum" == self.reduction:
            jsd = torch.sum(jsd)
        elif "batchmean" == self.reduction:
            jsd = torch.sum(jsd, dim=1)
            jsd = torch.mean(jsd)

        return jsd
