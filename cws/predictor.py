from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import transformers as tr
from tqdm import tqdm

from dataset import Dataset
from models import ChineseSegmenter


class Predictor:
    def __init__(self, model_path):
        self.bies_dict = {1: "B", 2: "I", 3: "E", 4: "S"}
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.model = ChineseSegmenter.load_from_checkpoint(model_path)
        print(self.model)
        self.tokenizer = tr.BertTokenizer.from_pretrained(
            self.model.hparams.language_model, tokenize_chinese_chars=True
        )
        self.model_max_length = self.tokenizer.model_max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def predict(self, input_path: str, output_path: str):
        test_file = Dataset.read_dataset(input_path)
        with open(output_path, "w") as out_file:
            for line in tqdm(test_file):
                words_pred = self.prediction_generator(line)
                out_file.write("".join(words_pred).strip() + "\n")

    def prediction_generator(self, line: List[str]):
        line = [c for c in line]
        prediction = self._get_predictions(line[: self.model_max_length - 10])
        if len(line) > self.tokenizer.model_max_length - 10:
            prediction_left = self._get_predictions(line[self.model_max_length - 10 :])
            prediction = np.concatenate([prediction, prediction_left])
        bies_pred = [self.bies_dict[p + 1] for p in prediction]
        words_pred = [
            c if bies_pred[i] in ("B", "I") else c + " " for i, c in enumerate(line)
        ]
        return words_pred

    def _get_predictions(self, line):
        example = self.tokenizer.encode_plus(
            line,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        example = {k: v.to(self.device) for k, v in example.items()}
        prediction = self.model(example, training=False)[0]
        prediction = self.softmax_fn(prediction).cpu().data.numpy()
        prediction = prediction[1:-1, 1:].argmax(axis=-1)
        return prediction


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_file", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("model_path", help="The path of the model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = Predictor(args.model_path)
    predictor.predict(args.input_path, args.output_path)
