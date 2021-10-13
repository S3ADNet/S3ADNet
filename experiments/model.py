from __future__ import annotations
from abc import ABCMeta
from argparse import ArgumentParser
import os.path
import sys
from typing import Any, Optional, Union, Callable


import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torchmetrics import functional as FM


dirname = os.path.dirname(__file__)                 # nopep8
sys.path.append(os.path.join(dirname, '..'))        # nopep8

from s3adnet import S3ADNet, SeNTXentLoss, CARELoss, SequentialKLDiv
from s3adnet.pooling import Pooling


def _set_activation(name: str) -> Callable:
    if "tanh" == name:
        def activation(): return nn.Tanh()
    elif "relu" == name:
        def activation(): return nn.ReLU(True)
    elif "leaky" == name:
        def activation(): return nn.LeakyReLU(0.2, True)
    else:
        raise ValueError(f"Unknown activation: {name}")

    return activation


class Base(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 feature_layers: list[nn.Module],
                 appending: Union[list[nn.Module], None],
                 feature_size: int,
                 embedding_size: int,
                 bias: bool,
                 embedding_noise: bool,
                 feature_drop: float,
                 activation: str,
                 **kwargs) -> None:

        super().__init__()

        self.embedding_noise = embedding_noise
        self.feature_drop = feature_drop

        self.feature = self._feature(feature_layers, activation, appending)
        self.embedding = nn.Linear(feature_size, embedding_size, bias)
        if embedding_noise:
            self.perturbation = nn.Linear(feature_size, 1, bias)

    def _feature(self, feature_layers: list[nn.Module],
                 activation: str,
                 appending: Optional[list[nn.Module]] = None):
        activation = _set_activation(activation)

        layers = []
        for layer in feature_layers:
            layers.append(layer)
            layers.append(activation())
            if self.feature_drop != 0.:
                layers.append(nn.Dropout(self.feature_drop))

        if appending is not None:
            layers.extend(appending)

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        if self.training and self.embedding_noise:
            x, sigma = self.embedding(x), self.perturbation(x)
            x += sigma * torch.randn_like(x)
        else:
            x = self.embedding(x)

        return x

    @staticmethod
    def add_model_specific_args(
            parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group('base')
        parser.add_argument("--embedding-noise", action='store_true',
                            help="whether adds gaussian noise to embeddings")
        parser.add_argument("--feature-drop", type=float, default=0.,
                            help="the dropout probability in feature layers")
        return parent_parser


class AdaptiveConcat1d(nn.Module):

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()

        self.mp = nn.AdaptiveAvgPool1d(output_size)
        self.ap = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, input: Tensor) -> Tensor:
        return torch.cat([self.mp(input), self.ap(input)], dim=1)


class BaseConv1d(Base):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 1,
                 bias: bool = True,
                 **kwargs) -> None:

        layers = [
            nn.Conv1d(in_channels, out_channels,
                      kernel_size, padding=padding, bias=bias),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size, padding=padding, bias=bias),
        ]
        appending = [
            AdaptiveConcat1d(1),
            nn.Flatten()
        ]

        super().__init__(feature_layers=layers,
                         appending=appending,
                         feature_size=2*out_channels,
                         bias=bias,
                         **kwargs)


class BaseMLP(Base):

    def __init__(self, input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 activation: str = "leaky",
                 **kwargs) -> None:

        layers = [
            nn.Linear(input_size, hidden_size, bias),
            nn.Linear(hidden_size, hidden_size, bias),
        ]

        super().__init__(feature_layers=layers,
                         appending=None,
                         feature_size=hidden_size,
                         bias=bias,
                         activation=activation,
                         **kwargs)


_optimizer = {
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'adam': optim.Adam,
    'adamw': optim.AdamW
}


class S3ADNetModel(pl.LightningModule):

    def __init__(self,
                 base: nn.Module,
                 input_size: int,
                 seq_len: int,
                 num_concepts: int,
                 temperature: float,
                 adaptation: Union[str, Callable[[int], float]],
                 lookahead_ratio: float,
                 penalty: float,
                 adaptation_coef: float,
                 lambda_sntx: float,
                 lambda_skld: float,
                 lambda_care: float,
                 warmup_epochs: int,
                 clip_grad_value: float,
                 lr_base: float,
                 lr_gate: float,
                 lr_finetune: float,
                 momentum: float,
                 weight_decay: float,
                 concept_pool: Union[str, Pooling] = 'absmax',
                 context_pool: Union[str, Pooling] = 'robustavg',
                 optimizer: str = 'rmsprop',
                 pred_steps: int = 1,
                 bulk_size: int = 256,
                 input_augmented: bool = False,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.input_augmented = input_augmented

        self.base = base
        self.model = S3ADNet(self.base,
                             input_size,
                             num_concepts=num_concepts,
                             concept_pool=concept_pool,
                             context_pool=context_pool,
                             compute_gates=True,
                             output_probs=True,
                             bulk_size=bulk_size)

        # Losses
        self.skld = SequentialKLDiv()
        # self.sjsd = SequentialJSDiv()
        self.sntxl = SeNTXentLoss(temperature=temperature)

        self.carel = CARELoss(seq_len,
                              adaptation=adaptation,
                              lookahead=int(seq_len * lookahead_ratio),
                              penalty=penalty,
                              adaptation_coef=adaptation_coef)

        self.pred_loss = CARELoss(seq_len,
                                  adaptation=adaptation,
                                  lookahead=1,
                                  penalty=1.,
                                  adaptation_coef=adaptation_coef)

        self.lambda_sntx = lambda_sntx
        self.lambda_care = lambda_care

        if input_augmented is False:
            self.lambda_skld = lambda_skld

        self.warmup_epochs = warmup_epochs

        self.clip_grad_value = clip_grad_value

        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.optimizer_name = optimizer

        self.lr_base = lr_base
        self.lr_gate = lr_gate
        self.lr_finetune = lr_finetune
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.automatic_optimization = False

    @staticmethod
    def add_model_specific_args(parent_parser:
                                ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group('s3adnet')
        parser.add_argument("--seq_len", type=int, default=4,
                            help="the length of a sequence")
        parser.add_argument("--num_concepts", type=int,
                            default=8, help="the number of concepts")
        parser.add_argument("--temperature", type=int, default=0.1,
                            help="temperature parameter for Re-Sent^2")
        parser.add_argument("--adaptation", type=str, default="sqrt",
                            help="the adaptation function for CARE which can be one of 'sqrt', 'log1p', and 'exp'")
        parser.add_argument("--lookahead_ratio", type=float, default=0.5,
                            help="to calculate how many steps to look ahead for the sequential relationships")
        parser.add_argument("--penalty", type=float, default=0.1,
                            help="the penalty to the entropy maximization")
        parser.add_argument("--adaptation_coef", type=float, default=0.5,
                            help="the coefficient for the adaptation function")
        parser.add_argument("--lambda_sntx", type=float,
                            default=1., help="the weight of Re-Sent^2")
        parser.add_argument("--lambda_skld", type=float, default=0.5,
                            help="the weight of the regularization of Re-Sent^2")
        parser.add_argument("--lambda_care", type=float,
                            default=5., help="the weight of CARE")
        parser.add_argument("--warmup_epochs", type=int, default=20,
                            help="the epoch numbers for the warm-up learning")
        parser.add_argument("--clip_grad_value",
                            type=float, default=10., help=" ")
        parser.add_argument("--lr_base", type=float, default=0.1,
                            help="the learning rate for the encoder")
        parser.add_argument("--lr_gate", type=float, default=0.1,
                            help="the learning rate for the MCC layer")
        parser.add_argument("--lr_finetune", type=float,
                            default=1e-4, help="the learning rate for fine-tuning")
        parser.add_argument("--momentum", type=float, default=0.9, help=" ")
        parser.add_argument("--weight_decay", type=float,
                            default=1e-4, help=" ")
        parser.add_argument("--pred_steps", type=int,
                            default=1, help="the stride of the sliding window on datasets in prediction")

        return parent_parser

    def forward(self, input: Tensor) -> Any:
        return self.model(input)

    def criterion_contrast(self, proj1: Tensor, proj2: Tensor) -> Tensor:
        loss = self.sntxl(proj1, proj2)
        self.log('loss_con', loss, on_step=True)

        if self.input_augmented is False:
            loss += self.criterion_augment(proj1, proj2)
        return loss

    def criterion_augment(self, proj1: Tensor, proj2: Tensor) -> Tensor:
        loss = self.lambda_skld / self.lambda_sntx * \
            self.skld(proj1, proj2)
        self.log('loss_aug', loss, on_step=True)
        return loss

    def criterion_relate(self, proj1: Tensor, proj2: Tensor,
                         pred1: Tensor, pred2: Tensor) -> Tensor:
        loss = 0.5 * (self.carel(proj1, pred1) + self.carel(proj2, pred2))
        self.log('loss_rel', loss, on_step=True)
        return loss

    def criterion(self, proj1: Tensor, proj2: Tensor,
                  pred1: Tensor, pred2: Tensor) -> Tensor:
        loss = self.lambda_sntx * self.criterion_contrast(
            proj1, proj2)
        loss += self.lambda_care * self.criterion_relate(
            proj1, proj2, pred1, pred2)
        return loss

    def training_step(self, batch: Tensor,
                      batch_idx: int) -> dict[str, Any]:
        opt_base, opt_gate = self.optimizers()

        opt_base.zero_grad()
        opt_gate.zero_grad()

        input = batch

        if self.input_augmented is True:
            input1, input2 = input
        else:
            input1 = input.clone()
            input2 = input

        if self.current_epoch < self.warmup_epochs:
            self.model.compute_gates = False
            proj1 = self(input1)
            proj2 = self(input2)
            loss = self.criterion_contrast(proj1, proj2)
        else:
            self.model.compute_gates = True
            proj1, pred1 = self(input1)
            proj2, pred2 = self(input2)

            opt_base.param_groups[0]['lr'] = self.lr_finetune
            loss = self.criterion(proj1, proj2, pred1, pred2)

        self.manual_backward(loss)
        nn.utils.clip_grad_value_(self.model.parameters(),
                                  self.clip_grad_value)

        opt_base.step()
        opt_gate.step()

        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)

    def predict_step(self, batch: Tensor,
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None) -> Tensor:
        if isinstance(batch, tuple) or isinstance(batch, list):
            input, target = batch
        else:
            input = batch
        _, pred = self(input)
        return pred.data.cpu()

    def reduce_results(self, results: list[Tensor],
                       has_losses: bool = False,
                       has_targets: bool = False) -> Union[
                           Tensor, tuple[Tensor, Tensor],
                           tuple[Tensor, Tensor, Tensor]]:
        seq_len = self.seq_len
        steps = self.pred_steps

        if has_targets and has_losses:
            preds, losses, targets = zip(*results)
            preds = torch.cat(preds)
            losses = torch.stack(losses)
            targets = torch.cat(targets)

        elif has_losses:
            preds, losses = zip(*results)
            preds = torch.cat(preds)
            losses = torch.stack(losses)

        elif has_targets:
            preds, targets = zip(*results)
            preds = torch.cat(preds)
            targets = torch.cat(targets)

        else:
            preds = torch.cat(results)

        total = (len(preds) - 1) * steps + seq_len
        results = torch.zeros(total)

        if has_targets:
            labels = torch.zeros(total, dtype=torch.int)

        iterator = zip(preds, targets) if has_targets else preds

        for i, item in enumerate(iterator):
            j = i * steps

            if has_targets:
                pred, target = item
                labels[j:j+seq_len] = target
            else:
                pred = item

            results[j:j+seq_len] = torch.maximum(results[j:j+seq_len], pred)

        results = results.nan_to_num_(1.0)

        if has_losses and has_targets:
            return results, losses, labels
        elif has_losses:
            return results, losses
        elif has_targets:
            return results, labels

        return results

    def validation_step(self, batch: Tensor,
                        batch_idx: int) -> tuple[Tensor, Tensor]:
        self.model.compute_gates = True
        input = batch
        proj, pred = self(input)
        loss = self.pred_loss(proj, pred)

        return pred.data.cpu(), loss.data.cpu()

    def test_step(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        return self.validation_step(*args, **kwargs)

    def _log_evalutation(self, prefix: str,
                         preds: Tensor, targets: Tensor) -> None:

        targets = targets.view_as(preds)

        metrics = dict(f1=FM.f1(preds, targets))
        metrics["precision"], metrics["recall"] = FM.precision_recall(
            preds, targets)
        metrics = {f"{prefix}_{name}": value
                   for name, value in metrics.items()}
        self.log_dict(metrics)

    def validation_epoch_end(self, outputs: list[Tensor]) -> None:
        preds, losses = self.reduce_results(
            outputs, has_losses=True)
        targets = self.trainer.datamodule.y_val[:len(preds)]
        self.log("val_avg_loss", losses.mean())
        self._log_evalutation("val", preds, targets)

    def test_epoch_end(self, outputs: list[Tensor]) -> None:
        preds, losses = self.reduce_results(outputs, has_losses=True)
        targets = self.trainer.datamodule.y_test[:len(preds)]
        self.log("test_avg_loss", losses.mean())
        self._log_evalutation("test", preds, targets)

    def configure_optimizers(self) -> list[Optimizer]:
        optimizer = _optimizer[self.optimizer_name]

        kwargs = {}
        if self.optimizer_name in {'sgd', 'rmsprop', 'adam', 'adamw'}:
            kwargs['weight_decay'] = self.weight_decay

        if self.optimizer_name in {'sgd', 'rmsprop'}:
            kwargs['momentum'] = self.momentum

        opt_base = optimizer(self.model.base.parameters(),
                             lr=self.lr_base, **kwargs)

        opt_gate = optimizer(self.model.gate.parameters(),
                             lr=self.lr_gate, **kwargs)

        return [opt_base, opt_gate]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        # from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(
            limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
