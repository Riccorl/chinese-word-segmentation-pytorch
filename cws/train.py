import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
    model_path = os.path.join(
        hparams.model_path,
        "{epoch:02d}-{val_loss:.2f}" + hparams.language_model.split("/")[-1],
    )
    print("Save checkponts in:", model_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_top_k=5,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    model = ChineseSegmenter(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)
