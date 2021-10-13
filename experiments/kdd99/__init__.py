from __future__ import annotations

import os.path
import sys
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import torch
from torch.utils.data import DataLoader


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # nopep8

from s3adnet.utils import sliding_window


cols_to_norm = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
                "hot", "num_failed_logins", "num_compromised", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "count", "srv_count",
                "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]


class KDD99DataModule(pl.LightningDataModule):

    def __init__(self, data_path: str, batch_size=1, seq_len=16,
                 scale='robust', stride=1, val_batch_size=1024, **kwargs):
        super().__init__()

        self.data_path = os.path.expanduser(data_path)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride

        self.scale = scale
        self.scaler = None

        self.val_batch_size = val_batch_size
        self.num_workers = kwargs.get("num_workers", 0)

    def setup(self, stage: Optional[str]) -> None:

        train_df = pd.read_pickle(self.data_path)

        half_train = train_df.shape[0] // 2
        x_train, y_train = _to_xy(train_df.loc[:half_train-1])
        x_val, y_val = _to_xy(train_df.loc[half_train:])

        if self.scale:
            if "minmax" == self.scale:
                self.scaler = MinMaxScaler()
            elif "robust" == self.scale:
                self.scaler = RobustScaler()
            elif "standard" == self.scale:
                self.scaler = StandardScaler()

            x_train = self.scaler.fit_transform(x_train)
            x_val = self.scaler.transform(x_val)

        self.x_train = _to_tensor(x_train)
        self.y_train = _to_tensor(y_train).view(-1)
        self.x_val = _to_tensor(x_val)
        self.y_val = _to_tensor(y_val).view(-1)

    def train_dataloader(self) -> DataLoader:
        train_set = sliding_window(self.x_train,
                                   self.seq_len, self.stride)

        return DataLoader(train_set,
                          batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:

        val_set = sliding_window(self.x_val,
                                 self.seq_len, self.stride)

        return DataLoader(val_set,
                          batch_size=self.val_batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        train_set = sliding_window(self.x_train,
                                   self.seq_len, self.stride)

        train_dataloader = DataLoader(train_set,
                                      batch_size=self.val_batch_size,
                                      shuffle=False, num_workers=self.num_workers)

        return [train_dataloader,
                self.val_dataloader()]


def _to_xy(df):
    cols = list(df.columns)
    cols.remove("label")
    x = df[cols].values.astype(np.float32)
    y = df["label"].values.flatten().astype(np.int64)
    return x, y


def _to_tensor(arr):
    return torch.from_numpy(arr).contiguous()
