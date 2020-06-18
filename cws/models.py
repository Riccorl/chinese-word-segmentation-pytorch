import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers as tr
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import Dataset


class ChineseSegmenter(pl.LightningModule):
    def __init__(
        self,
        input_file: str,
        hidden_size: int = 512,
        num_layers: int = 2,
        max_length: int = 200,
        batch_size: int = 32,
        language_model: str = "bert-base-chinese",
    ):
        super(ChineseSegmenter, self).__init__()

        # params
        self.lr = 0.001
        self.batch_size = batch_size
        # pl code
        self.batch_size = batch_size
        self.data = self._load_data(input_file, language_model, max_length)
        self.train_set, self.val_set = self._split_data(self.data)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # model definition
        # config = tr.BertConfig.from_pretrained(
        #     language_model, output_hidden_states=True
        # )
        self.lmodel = tr.BertModel.from_pretrained(
            language_model, output_hidden_states=True
        )
        self.lstms = nn.LSTM(
            self.lmodel.config.hidden_size * 4,
            hidden_size,
            num_layers=num_layers,
            dropout=0.4,
            bidirectional=True,
        )
        self.dropouts = nn.Dropout(0.5)
        self.output = nn.Linear(hidden_size * 2, 5)

    def forward(self, inputs, **kwargs):
        x = self.lmodel(
            inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
        )[2][-4:]
        # x = torch.stack(x, dim=0).sum(dim=0)
        x = torch.cat(x, dim=-1)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
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
            "batch_size": self.batch_size,
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
