import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models import ChineseSegmenter


def parse_args():

    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ChineseSegmenter.add_model_specific_args(parser)
    hparams = parser.parse_args()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )
    print(hparams.model_path)
    model_path = os.path.join(
        hparams.model_path,
        hparams.language_model.split("/")[-1] + "_{epoch:02d}_{val_loss:.2f}.ckpt",
    )
    print("Save checkponts in:", model_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path, save_top_k=5, verbose=True
    )
    model = ChineseSegmenter(hparams)
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        default_root_dir=os.path.join(os.getcwd(), "logs"),
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)
