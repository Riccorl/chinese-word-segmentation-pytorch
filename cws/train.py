import os
import sys
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import utils
from models import ChineseSegmenter, ChineseSegmenterLSTM


def main():
    # fix until transformers fixes its bug
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser()
    parser.add_argument("--run", help="run of the model", default=1)
    parser.add_argument("--dataset", help="dataset name")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ChineseSegmenter.add_model_specific_args(parser)
    parser = ChineseSegmenterLSTM.add_model_specific_args(parser)
    hparams = parser.parse_args()
    print("Device:", device)
    print("Dataset:", hparams.dataset)
    if hparams.embeddings_file:
        print("Model: lstm")
        model = ChineseSegmenterLSTM(hparams)
        model_path = os.path.join(
            hparams.model_path,
            "lstm",
            hparams.dataset,
            "run_" + hparams.run,
            "model_{epoch:02d}_{val_loss:.4f}",
        )
    else:
        print("Model: transformer")
        model = ChineseSegmenter(hparams)
        model_path = os.path.join(
            hparams.model_path,
            hparams.language_model.split("/")[-1],
            hparams.dataset,
            "run_" + hparams.run,
            "model_{epoch:02d}_{val_loss:.4f}",
        )

    print("Save checkponts in:", model_path)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path, save_top_k=5, verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=4, verbose=True, mode="min"
    )
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        default_root_dir=os.path.join(os.getcwd(), "logs"),
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
    )

    if not model.hparams.lr:
        # Run learning rate finder
        model.hparams.lr = 0.001  # default adam, if None, crash
        print("No learning rate provided, finding optimal lr")
        lr_finder = trainer.lr_find(model)
        optim_lr = lr_finder.suggestion()
        print()
        print("Optimized learning rate", optim_lr)
        print()
        model.hparams.lr = optim_lr
    trainer.fit(model)
    print("best model:", checkpoint_callback.best_model_path)
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    start = time.time()
    best_model_path = main()
    end = time.time()
    print("Total training time:", utils.timer(start, end))
    with open("predictions/best_model_path.txt", "w") as fh:
        fh.write(best_model_path)
