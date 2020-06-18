from typing import List, Dict, Sequence, Tuple

import torch
import transformers as tr


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path: str,
        language_model: str = "bert-base-chinese",
        max_length: int = 200,
    ):
        self.bies_dict = {"B": 1, "I": 2, "E": 3, "S": 4}
        self.file_path = file_path
        print(file_path)
        # self.tokenizer = tr.AutoTokenizer.from_pretrained(language_model)
        self.tokenizer = tr.BertTokenizer.from_pretrained(
            language_model, tokenize_chinese_chars=True
        )
        self.max_length = max_length
        self.features, self.labels = self.process_data()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Select sample
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def process_data(self):
        features, labels = self.load_files()
        x = [self._process_text(f, self.max_length) for f in features]
        y = [self._convert_labels(f, self.max_length) for f in labels]
        return x, y

    def _process_text(self, text: str, max_length: int) -> Dict[str, Sequence[int]]:
        """
        Preprocess the text according to the language model
        tokenizer
        :param text: text to preprocess
        :return: a dictionaty containg the data for the model
        """
        inputs = self.tokenizer.encode_plus(
            [c for c in text],
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=max_length,
            pad_to_max_length=True,
        )
        return inputs.data

    def _convert_labels(self, bies_line: str, max_length: int) -> Sequence[int]:
        """
        Convert a BIES line to integer based label for the model
        :param bies_line: BIES line in input
        :return: the same line with numerical labels
        """
        converted_line = [self.bies_dict[c] for c in bies_line[: max_length - 2]]
        converted_line = [0] + converted_line + [0]
        if len(converted_line) < max_length:
            converted_line += [0] * (max_length - len(converted_line))
        return converted_line

    def load_files(self) -> Tuple[List[str], List[str]]:
        """
        Load features and labels from files
        :return: list of features and labels
        """
        filename, _, ext = self.file_path.rpartition(".")
        features_file = filename + "_nospace." + ext
        labels_file = filename + "_bies." + ext
        print("Features file:", features_file)
        print("Labels file:", labels_file)
        features = self.read_dataset(features_file)
        labels = self.read_dataset(labels_file)
        return features, labels

    @staticmethod
    def read_dataset(filename: str) -> List[str]:
        """
        Read the dataset line by line.
        :param filename: file to read
        :return: a list of lines
        """
        with open(filename, encoding="utf8") as file:
            f = (line.strip() for line in file)
            return [line for line in f if line]

    @staticmethod
    def generate_batch(batch):
        input_ids = torch.tensor([b[0]["input_ids"] for b in batch])
        attention_mask = torch.tensor([b[0]["attention_mask"] for b in batch])
        token_type_ids = torch.tensor([b[0]["token_type_ids"] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return features, labels
