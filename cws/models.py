from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import transformers as tr
from torch.utils.data import DataLoader

from dataset import Dataset


class ChineseSegmenter(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(ChineseSegmenter, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # model definition
        self.lmodel = tr.BertModel.from_pretrained(
            self.hparams.language_model, output_hidden_states=True
        )
        self.lstms = nn.LSTM(
            self.lmodel.config.hidden_size * 4,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            dropout=0.4,
            bidirectional=True,
        )
        self.dropouts = nn.Dropout(0.5)
        self.output = nn.Linear(self.hparams.hidden_size * 2, 5)

    def forward(self, inputs, **kwargs):
        x = self.lmodel(
            inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
        )[2][-4:]
        if self.hparams.mode == "concat":
            x = torch.cat(x, dim=-1)
        elif self.hparams.mode == "sum":
            x = torch.stack(x, dim=0).sum(dim=0)
        else:
            raise ValueError('Mode not supported, chose between "concat" and "sum"')
        x, _ = self.lstms(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.view(-1, 5), y.view(-1))
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.view(-1, 5), y.view(-1))
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        results = {"progress_bar": logs}
        return results

    def prepare_data(self):
        data = self._load_data(
            self.hparams.input_file, self.hparams.language_model, self.hparams.max_len
        )
        self.train_set, self.val_set = self._split_data(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        params = self._get_loader_params()
        return DataLoader(self.train_set, **params)

    def val_dataloader(self):
        params = self._get_loader_params(False)
        return DataLoader(self.val_set, **params)

    def _get_loader_params(self, train=True):
        return {
            "batch_size": self.hparams.batch_size,
            "shuffle": train,
            "num_workers": 3,
            "collate_fn": Dataset.generate_batch,
        }

    @staticmethod
    def _load_data(input_file: str, language_model: str, max_length: int):
        return Dataset(input_file, language_model=language_model, max_length=max_length)

    @staticmethod
    def _split_data(data):
        train_len = int((90 * len(data)) // 100.0)
        val_len = len(data) - train_len
        print("Train size:", train_len)
        print("Val size:", val_len)
        return torch.utils.data.random_split(data, [train_len, val_len])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--input_file", help="The path of the input file", required=True
        )
        parser.add_argument(
            "--hidden_size", help="LSTM hidden size", type=int, default=512
        )
        parser.add_argument(
            "--num_layers", help="number of LSTM layers", type=int, default=2
        )
        parser.add_argument(
            "--max_len", help="max sentence length", type=int, default=150
        )
        parser.add_argument(
            "--batch_size", help="size of the batch", type=int, default=32
        )
        parser.add_argument(
            "--lr", help="starting learning rate", type=float, default=0.01
        )
        parser.add_argument(
            "--mode",
            help="bert output mode",
            default="concat",
            choices=["concat", "sum"],
        )
        parser.add_argument(
            "--language_model",
            help="language model to use",
            default="bert-base-chinese",
        )
        return parser
