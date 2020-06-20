import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models import ChineseSegmenter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ChineseSegmenter.add_model_specific_args(parser)
    hparams = parser.parse_args()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )
    model_path = os.path.join(
        hparams.model_path,
        hparams.language_model.split("/")[-1],
        "model_{epoch:02d}_{val_loss:.2f}",
    )
    print("Save checkponts in:", model_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path, save_top_k=5, verbose=True
    )
    # Run learning rate finder
    model = ChineseSegmenter(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        default_root_dir=os.path.join(os.getcwd(), "logs"),
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )

    lr_finder = trainer.lr_find(model)
    optim_lr = lr_finder.suggestion()
    print("Optimized learning rate", optim_lr)
    print()
    model.hparams.lr = optim_lr
    trainer.fit(model)
    print("best model:", checkpoint_callback.best_model_path)
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    best_model_path = main()
    # calculate
    with open("predictions/best_model_path.txt", "w") as fh:
        fh.write(best_model_path)
