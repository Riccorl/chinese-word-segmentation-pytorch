from argparse import ArgumentParser
from typing import List

import torch
from tqdm import tqdm

from dataset import Dataset


class Predictor:
    def __init__(self, model_path):
        self.bies_dict = {1: "B", 2: "I", 3: "E", 4: "S"}
        self.softmax_fn = torch.nn.Softmax()
        self.model = torch.load(model_path)
        self.tokenizer = self.model.data.tokenizer
        self.model_max_length = self.tokenizer.model_max_length

    def predict(self, input_path: str, output_path: str):
        bies_path = output_path + "_bies.utf8"
        words_path = output_path + "_words.utf8"
        test_file = Dataset.read_dataset(input_path)
        bies_ctx = open(bies_path, "w")
        words_ctx = open(words_path, "w")
        with bies_ctx as bies_file, words_ctx as words_file:
            bies_pred, words_pred = self.prediction_generator(test_file)
            bies_file.write("".join(bies_pred) + "\n")
            words_file.write("".join(words_pred).strip() + "\n")

    def prediction_generator(self, test_file: List[str]):
        for line in tqdm(test_file):
            line = [c for c in line]
            prediction = self._get_predictions(line[: self.model_max_length])
            if len(line) > self.tokenizer.model_max_length:
                prediction += self._get_predictions(line[self.model_max_length :])
            bies_pred = [self.bies_dict[p + 1] for p in prediction]
            words_pred = [
                c if bies_pred[i] in ("B", "I") else c + " " for i, c in enumerate(line)
            ]
            yield bies_pred, words_pred

    def _get_predictions(self, line):
        example = self.tokenizer.encode_plus(
            line,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        example = {k: v.to(self.model.device) for k, v in example.items()}
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
