from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, device
from torch.nn import Module


def get_model_device(model: Module,
                     default: Optional[device] = None) -> Optional[device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return default


def bulk_forward(model: Module, input: Tensor, bulk_size: int = None,
                 default_device: Optional[device] = None) -> Tensor:

    batch_size, seq_len, *input_dims = input.size()
    device = get_model_device(model, default_device)

    if bulk_size:
        input = input.view(batch_size * seq_len, *input_dims)
        splits = torch.split(input, bulk_size)
        if len(input) > 0:
            input = torch.cat(
                [model(split.to(device)) for split in splits], dim=0)
        else:
            input = model(input.to(device))

        input = input.view(batch_size, seq_len, *input.size()[1:])
    else:
        chunks = torch.chunk(input, seq_len, dim=1)
        input = torch.stack(
            [model(chunck.squeeze(1).to(device)) for chunck in chunks], dim=1)

    return input


def _iter_reversed_cumprods_with_one(s):
    p = 1
    yield p
    for n in reversed(s):
        p *= n
        yield p


def _cumprods_with_one(s):
    return tuple(_iter_reversed_cumprods_with_one(s))[::-1]


def sliding_window(inp: Tensor, win_len: int, step=1) -> Tensor:
    r"""Generates new tensor with copied values by sliding a window along the first dim.

    Parameters
    ----------
    inp : Tensor
        :math:`(L, M_1, M_2, ...)`
    win_len : int
        sliding window size :math:`w`
    step : int, optional
        sliding step size :math:`s`, by default 1

    Returns
    -------
    Tensor
        :math:`\right(\lfloor\frac{(L - w)}{s}\rfloor + 1, w, M_1, M_2, ...\left)`
    """
    inp_size = inp.size()
    inp_len = inp_size[0]
    feat_size = inp_size[1:]

    stride = _cumprods_with_one(feat_size)      # (M_1*M_2*..., M_2*..., ..., 1)      # nopep8
    stride = (stride[0] * step,) + stride       # (s*M_1*M_2*..., M_1*M_2*..., M_2*..., ..., 1)   # nopep8
    out_size = ((inp_len - win_len) // step + 1, win_len) + feat_size

    view = torch.as_strided(inp, out_size, stride)
    return view.clone()
