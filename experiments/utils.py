

from __future__ import annotations
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import random

import numpy as np
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.tensorboard import SummaryWriter

from .model import Base, S3ADNetModel


def init_experiment(args: Namespace) -> None:
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def get_default_parser(*args, **kwargs) -> ArgumentParser:
    parser = ArgumentParser(
        *args,
        formatter_class=ArgumentDefaultsHelpFormatter,
        **kwargs)

    parser = Trainer.add_argparse_args(parser)
    parser = Base.add_model_specific_args(parser)
    parser = S3ADNetModel.add_model_specific_args(parser)

    group = parser.add_argument_group("tracking")
    group.add_argument(
        "--checkpoint_dir",
        type=str,
        default="~/checkpoints",
        help=" "
    )
    group.add_argument(
        "--tb_dir",
        type=str,
        default="~/tb_logs",
        help="TensorBoard path"
    )
    group.add_argument(
        "--experiment",
        type=str,
        help=" "
    )
    group.add_argument(
        "--tag_name",
        type=str,
        default="Default",
        help="for version tagging in TensorBoard"
    )

    group = parser.add_argument_group('environment')
    group.add_argument(
        "--seed",
        type=int,
        default=777,
        help=" "
    )
    group.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="for dataloaders"
    )

    group.add_argument(
        "--bulk_size",
        type=int,
        default=256,
        help="the number of data points for parallel encoding"
    )

    group.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help=" "
    )

    group.add_argument(
        "--val_batch_size",
        type=int,
        default=256,
        help=" "
    )

    return parser


class TensorboardHistogramLogging(Callback):

    def __init__(self, every_n_steps=500,
                 reduction=torch.norm) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.reduction = reduction

    def on_after_backward(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        global_step = pl_module.trainer.global_step
        reduction_name = self.reduction.__name__

        if global_step % self.every_n_steps == 0:

            writer: SummaryWriter = pl_module.logger.experiment
            for name, params in pl_module.named_parameters():
                if params.grad is None:
                    continue

                grad = params.grad.detach()
                weight = params.data.detach()

                try:
                    writer.add_histogram(
                        f"grads/{name}", grad, global_step)
                    writer.add_histogram(
                        f"weights/{name}", weight, global_step)
                    writer.add_scalar(
                        f"grads_{reduction_name}/{name}", self.reduction(grad), global_step)
                    writer.add_scalar(
                        f"weights_{reduction_name}/{name}", self.reduction(weight), global_step)

                except ValueError:
                    continue

                finally:
                    grad = None
                    weight = None
