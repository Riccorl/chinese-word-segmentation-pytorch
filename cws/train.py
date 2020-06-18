from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from models import ChineseSegmenter


def parse_args():

    return parser.parse_args()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ChineseSegmenter.add_model_specific_args(parser)
    hparams = parser.parse_args()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
    )
    model = ChineseSegmenter(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams, early_stop_callback=early_stop_callback
    )
    trainer.fit(model)
