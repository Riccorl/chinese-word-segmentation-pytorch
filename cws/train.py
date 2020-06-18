from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from models import ChineseSegmenter


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_file", help="The path of the input file")
    parser.add_argument("--epochs", help="number of epochs", default=10, type=int)
    parser.add_argument(
        "--batch_size", help="size of the batch", default=32, type=int,
    )
    parser.add_argument(
        "--model_path", help="where to store checkpoints", required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    language_model = "clue/roberta_chinese_clue_tiny"
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
    )
    model = ChineseSegmenter(
        args.input_file,
        language_model=language_model,
        max_length=50,
        batch_size=args.batch_size,
    )
    trainer = pl.Trainer(
        early_stop_callback=early_stop_callback,
        gpus=1,
        max_epochs=args.epochs,
        default_root_dir=args.model_path,
        auto_lr_find=True,
    )
    trainer.fit(model)
