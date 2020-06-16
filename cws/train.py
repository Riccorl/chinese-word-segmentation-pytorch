from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.engine import (
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss, Fbeta, Precision, Recall, RunningAverage
from tqdm.auto import tqdm

from dataset import Dataset
from models import ChineseSegmenter


class Trainer:
    def __init__(
        self, input_file: str, language_model: str, batch_size: int, max_length: int
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        self.data = self._load_data(input_file, language_model, max_length)
        self.train_loader, self.val_loader = self._get_loaders(self.data, batch_size)
        self.model = self._load_model(language_model, max_length)

    def train(self, epochs: int):
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=self.data.tokenizer.pad_token_id)
        # metrics
        metrics = {
            "avg_loss": Loss(criterion),
            "avg_precision": Precision(average=True),
            "avg_recall": Recall(average=True),
            "avg_f1": Fbeta(beta=1, average=True),
        }
        # trainer
        trainer = create_supervised_trainer(
            self.model, optimizer, criterion, self.device
        )
        evaluator = create_supervised_evaluator(
            self.model, metrics=metrics, device=self.device
        )
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_training_results, evaluator
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_validation_results, evaluator
        )
        early_stopping = EarlyStopping(
            patience=5, score_function=self.score_fn, trainer=evaluator
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopping)
        self._setup_pbar(trainer)
        output = trainer.run(self.train_loader, max_epochs=epochs)

    @staticmethod
    def _setup_pbar(trainer):
        pbar = ProgressBar(
            persist=True,
            bar_format="",
            # event_name=Events.EPOCH_STARTED,
            # closing_event_name=Events.COMPLETED,
        )
        pbar.attach(trainer, ["loss"])

    def score_fn(self, engine):
        score = engine.state.metrics["avg_f1"]
        return score

    @staticmethod
    def _get_loaders(data, batch_size):
        # Parameters
        params = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
            "collate_fn": Dataset.generate_batch,
        }
        # split data
        train_len = int((90 * len(data)) // 100.0)
        val_len = len(data) - train_len
        print("Train size:", train_len)
        print("Val size:", val_len)
        train_set, val_set = torch.utils.data.random_split(data, [train_len, val_len])
        # load generators
        train_loader = torch.utils.data.DataLoader(train_set, **params)
        val_loader = torch.utils.data.DataLoader(val_set, **params)
        return train_loader, val_loader

    def log_training_results(self, trainer, evaluator):
        evaluator.run(self.train_loader)
        metrics = evaluator.state.metrics
        update_string = "Training Results - Epoch: {}  Avg f1: {:.2f} Avg loss: {:.2f}".format(
            trainer.state.epoch, metrics["avg_f1"], metrics["avg_loss"]
        )
        print(update_string)

    def log_validation_results(self, trainer, evaluator):
        evaluator.run(self.val_loader)
        metrics = evaluator.state.metrics
        update_string = "Validation Results - Epoch: {}  Avg f1: {:.2f} Avg loss: {:.2f}".format(
            trainer.state.epoch, metrics["avg_f1"], metrics["avg_loss"]
        )
        print(update_string)

    @staticmethod
    def _load_model(language_model, max_length):
        return ChineseSegmenter(language_model=language_model, max_length=max_length)

    @staticmethod
    def _load_data(input_file: str, language_model: str, max_length: int):
        return Dataset(input_file, language_model=language_model, max_length=max_length)


def train(input_file, batch_size, max_length, language_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Parameters
    params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,
        "collate_fn": Dataset.generate_batch,
    }
    epochs = 30
    # Load data
    data = Dataset(input_file, language_model=language_model, max_length=max_length)
    # split data
    train_len = int((90 * len(data)) // 100.0)
    val_len = len(data) - train_len
    print("Train size:", train_len)
    print("Val size:", val_len)
    train_set, val_set = torch.utils.data.random_split(data, [train_len, val_len])
    # load generators
    train_generator = torch.utils.data.DataLoader(train_set, **params)
    val_generator = torch.utils.data.DataLoader(val_set, **params)
    # train step
    model = ChineseSegmenter(
        language_model="clue/albert_chinese_small", max_length=max_length
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=data.tokenizer.pad_token_id).to(device)
    for epoch in range(epochs):
        # train step
        train_loop = tqdm(train_generator)
        train_loop.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        train_loss = train_step(criterion, device, train_loop, model, optimizer)
        # val step
        val_loop = tqdm(val_generator)
        val_loop.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        val_loss = val_step(model, criterion, device, train_loop)
        # print total loss
        epoch_loss = train_loss / len(train_generator)
        val_epoch_loss = val_loss / len(val_generator)
        print("train loss: {:.3}".format(epoch_loss))
        print("val loss: {:.3}".format(val_epoch_loss))


def train_step(criterion, device, loop, model, optimizer):
    # Training
    train_loss = 0.0
    for i, batch in enumerate(loop):
        # Transfer to GPU
        x, y = batch
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return train_loss


def val_step(model, criterion, device, loop):
    val_loss = 0.0
    for i, batch in enumerate(loop):
        # Transfer to GPU
        x, y = batch
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        # evaluate
        with torch.no_grad():
            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
        val_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return val_loss


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_file", help="The path of the input file")
    parser.add_argument("--epochs", help="number of epochs", default=10, type=int)
    parser.add_argument(
        "--batch_size", help="size of the batch", default=32, type=int,
    )
    parser.add_argument(
        "--checkpoint", help="where to store checkpoints", required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # train(args.input_file)
    language_model = "clue/roberta_chinese_clue_tiny"
    trainer = Trainer(args.input_file, language_model, args.batch_size, 50)
    trainer.train(args.epochs)
