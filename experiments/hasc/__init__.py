from __future__ import annotations

import os.path
import sys
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import scipy.io as sio
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # nopep8

from s3adnet.utils import sliding_window


class HASCDataModule(pl.LightningDataModule):

    def __init__(self, data_path, win_size, seq_len, batch_size=1,
                 stride=1, scale="standard", val_batch_size=256, **kwargs):
        super().__init__()

        self.data_path = os.path.expanduser(data_path)
        self.batch_size = batch_size
        self.win_size = win_size
        self.seq_len = seq_len
        self.stride = stride
        self.val_batch_size = val_batch_size

        self.scale = scale

        self.num_workers = kwargs.get("num_workers", 0)

    def setup(self, stage: Optional[str] = None) -> None:

        data = sio.loadmat(self.data_path)

        inputs = data['Y'].astype(np.float32)

        if self.scale:
            if "minmax" == self.scale:
                self.scaler = MinMaxScaler()
            elif "robust" == self.scale:
                self.scaler = RobustScaler()
            elif "standard" == self.scale:
                self.scaler = StandardScaler()

            inputs = self.scaler.fit_transform(inputs)

        N = inputs.shape[0]
        num_seq = N // self.win_size

        x_train = torch.from_numpy(inputs[:num_seq*self.win_size])
        x_train = x_train.reshape(num_seq, self.win_size, -1)
        x_train.transpose_(1, 2)

        self.x_train = x_train.contiguous()
        self.x_val = self.x_train.clone()

        targets = data['L'].astype(np.int64)

        y_val = torch.from_numpy(targets[:num_seq*self.win_size])
        y_val = y_val.reshape(num_seq, self.win_size)
        y_val = (y_val.sum(-1) >= 1).int()

        self.y_val = y_val

    def train_dataloader(self) -> DataLoader:
        train_set = sliding_window(
            self.x_train, self.seq_len, self.stride)

        return DataLoader(train_set, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        val_set = sliding_window(
            self.x_val, self.seq_len, self.stride)

        return DataLoader(val_set, batch_size=self.val_batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return self.val_dataloader()
