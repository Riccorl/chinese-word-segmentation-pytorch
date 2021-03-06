from argparse import ArgumentParser
from functools import partial

import gensim
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import transformers as tr
from torch.utils.data import DataLoader

from dataset import DatasetLM, DatasetLSTM


class ChineseSegmenter(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(ChineseSegmenter, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)

        self.criterion = nn.CrossEntropyLoss()
        # model definition
        config = tr.AutoConfig.from_pretrained(
            self.hparams.language_model, output_hidden_states=True
        )
        self.lmodel = tr.AutoModel.from_pretrained(
            self.hparams.language_model, config=config
        )
        if self.hparams.freeze:
            print("Freezeing lm weights")
            self.freeze_lm(self.lmodel)

        self.hparams.hidden_size = self.lmodel.config.hidden_size
        if self.hparams.bert_mode == "concat":
            self.hparams.hidden_size *= 4

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hparams.hidden_size, 5)

        # data
        self.data = self._load_data(
            self.hparams.input_file, self.hparams.language_model
        )
        self.train_set, self.val_set = self._split_data(self.data)

    def forward(self, inputs, *args, **kwargs):
        outputs = self.lmodel(**inputs)
        if self.hparams.bert_mode == "none":
            outputs = outputs[0]
        elif self.hparams.bert_mode == "concat":
            outputs = outputs[2][-4:]
            outputs = torch.cat(outputs, dim=-1)
        elif self.hparams.bert_mode == "sum":
            outputs = outputs[2][-4:]
            outputs = torch.stack(outputs, dim=0).sum(dim=0)
        if self.training:
            outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs

    def training_step(self, batch, *args):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._compute_active_loss(x, y, y_hat)
        return {"loss": loss}

    def validation_step(self, batch, *args):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._compute_active_loss(x, y, y_hat)
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
        if self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.95)
        else:
            parameters = self._optimizer_grouped_parameters()
            optimizer = tr.AdamW(parameters, lr=self.hparams.lr)
        return_values = [optimizer]
        if self.hparams.scheduler:
            num_warmup_steps = len(self.train_set) // self.hparams.batch_size * 3
            print("Warmpup steps:", num_warmup_steps)
            num_training_steps = (
                len(self.train_set) // self.hparams.batch_size * self.hparams.max_epochs
            )
            print("Training steps:", num_training_steps)
            scheduler = tr.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            return_values = ([optimizer], [scheduler])
        return return_values  # [optimizer], [scheduler]

    def _optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        no_params = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optimizer_grouped_parameters = [
            {"params": params, "weight_decay": 0.01},
            {"params": no_params, "weight_decay": 0.0},
        ]
        return (
            optimizer_grouped_parameters
            if self.hparams.optimized_decay
            else self.parameters()
        )

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
            "num_workers": 12,
            "pin_memory": True,
            "collate_fn": DatasetLM.generate_batch,
        }

    def _compute_active_loss(self, x, y, y_hat):
        active_loss = x["attention_mask"].view(-1) == 1
        active_logits = y_hat.view(-1, 5)
        active_labels = torch.where(
            active_loss,
            y.view(-1),
            torch.tensor(self.criterion.ignore_index).type_as(y),
        )
        return self.criterion(active_logits, active_labels)

    @staticmethod
    def _load_data(input_file: str, language_model: str):
        return DatasetLM(input_file, language_model=language_model)

    @staticmethod
    def _split_data(data):
        train_len = int((90 * len(data)) // 100.0)
        val_len = len(data) - train_len
        print("Train size:", train_len)
        print("Val size:", val_len)
        return torch.utils.data.random_split(data, [train_len, val_len])

    @staticmethod
    def freeze_lm(lmodel):
        for param in lmodel.base_model.parameters():
            param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--input_file", help="The path of the input file", required=True
        )
        parser.add_argument(
            "--max_len", help="max sentence length", type=int, default=150
        )
        parser.add_argument(
            "--batch_size", help="size of the batch", type=int, default=32
        )
        parser.add_argument("--lr", help="starting learning rate", type=float)
        parser.add_argument(
            "--bert_mode",
            help="bert output mode",
            default="none",
            choices=["none", "concat", "sum"],
        )
        parser.add_argument(
            "--language_model",
            help="language model to use",
            default="bert-base-chinese",
        )
        parser.add_argument(
            "--model_path", help="where to save model checkpoints", default="./models",
        )
        parser.add_argument(
            "--optimizer",
            help="which optimizer to use",
            default="sgd",
            choices=["sgd", "adamw"],
        )
        parser.add_argument(
            "--freeze", help="unfreeze embeddings ", action="store_true"
        )
        parser.add_argument(
            "--scheduler", help="use scheduler to control lr value", action="store_true"
        )
        parser.add_argument(
            "--optimized_decay",
            help="weight decay on subset of model params",
            action="store_true",
        )
        return parser


class ChineseSegmenterLSTM(ChineseSegmenter):
    def __init__(self, hparams, *args, **kwargs):
        super(ChineseSegmenter, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters(self.hparams)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        # model definition
        self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            self.hparams.embeddings_file, binary=True
        )
        self.embeddings = self._get_embeddings_layer(
            self.word_vectors.vectors, self.hparams.freeze
        )
        self.lstms = nn.LSTM(
            self.word_vectors.vector_size * 2,
            self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            dropout=0.5 if self.hparams.num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(self.hparams.hidden_size * 2, 5)

        # data
        self.data = self._load_data(self.hparams.input_file, self.word_vectors)
        self.train_set, self.val_set = self._split_data(self.data)

    def forward(self, inputs, *args, **kwargs):
        unigrams, bigrams = inputs
        outputs_unigrams = self.embeddings(unigrams)
        outputs_bigrams = self.embeddings(bigrams)
        outputs = torch.cat([outputs_unigrams, outputs_bigrams], dim=-1)
        outputs, _ = self.lstms(outputs)
        if self.training:
            outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs

    def training_step(self, batch, *args):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.view(-1, 5), y.view(-1))
        return {"loss": loss}

    def validation_step(self, batch, *args):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.view(-1, 5), y.view(-1))
        return {"val_loss": loss}

    def _get_loader_params(self, train=True):
        return {
            "batch_size": self.hparams.batch_size,
            "shuffle": train,
            "num_workers": 12,
            "pin_memory": True,
            "collate_fn": partial(DatasetLSTM.generate_batch, vocab=self.data.vocab),
        }

    @staticmethod
    def _get_embeddings_layer(weights, freeze: bool):
        # zero vector for pad, 1 in position 1
        # random vector for pad
        pad = np.random.rand(1, weights.shape[1])
        # mean vector for unknowns
        unk = np.mean(weights, axis=0, keepdims=True)
        weights = np.concatenate((pad, unk, weights))
        weights = torch.FloatTensor(weights)
        return nn.Embedding.from_pretrained(weights, padding_idx=0, freeze=freeze)

    @staticmethod
    def _load_data(input_file: str, word_vectors: gensim.models.word2vec.Word2Vec):
        return DatasetLSTM(input_file, word_vectors=word_vectors)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embeddings_file", help="The path of the embeddings file")
        parser.add_argument(
            "--hidden_size", help="LSTM hidden size", type=int, default=256
        )
        parser.add_argument(
            "--num_layers", help="number of LSTM layers", type=int, default=1
        )
        return parser
