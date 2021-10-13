import os.path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from experiments import S3ADNetModel, BaseMLP, BaseConv1d
from experiments.utils import (get_default_parser,
                               init_experiment,
                               TensorboardHistogramLogging)


if __name__ == "__main__":
    parser = get_default_parser()
    group = parser.add_argument_group('kddcup99')
    group.add_argument(
        "--kdd_path",
        type=str,
        default="./experiments/kdd99/data/kdd99_df.pkl.zip",
        help=" "
    )

    group = parser.add_argument_group('hasc')
    group.add_argument(
        "--hasc_path",
        type=str,
        default="./experiments/hasc/data/hasc-1.mat",
        help=" "
    )
    group.add_argument(
        "--win_size",
        type=int,
        default=4,
        help="the chunck size for the HASC dataset"
    )

    args = parser.parse_args()
    init_experiment(args)
    dict_args = vars(args)

    if args.experiment == 'kddcup99':
        from experiments.kdd99 import KDD99DataModule
        datamodule = KDD99DataModule(args.kdd_path, **dict_args)
        base = BaseMLP(121, 32, activation="leaky",
                       embedding_size=8, **dict_args)
        model = S3ADNetModel(base, 8, **dict_args)

    elif args.experiment == 'hasc':
        from experiments.hasc import HASCDataModule
        datamodule = HASCDataModule(args.hasc_path, **dict_args)
        base = BaseConv1d(3, 32, 3, padding=1, embedding_size=16,
                          activation="leaky", **dict_args)
        model = S3ADNetModel(base, 16, **dict_args)

    else:
        raise ValueError("`experiment` must be 'kddcup99' or 'hasc'!")

    logger = TensorBoardLogger(os.path.expanduser(args.tb_dir),
                               name=args.experiment,
                               version=args.tag_name)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.expanduser(args.checkpoint_dir),
        filename="model-{epoch:03d}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=-1,
        every_n_val_epochs=1)

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,
        callbacks=[
            checkpoint,
            TensorboardHistogramLogging()
        ])

    trainer.fit(model, datamodule=datamodule)
