from itertools import chain, tee
from typing import Dict, List, Sequence, Tuple

import gensim
import torch
import transformers as tr


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        self.bies_dict = {"B": 1, "I": 2, "E": 3, "S": 4}
        self.file_path = file_path
        print(file_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Select sample
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def process_data(self) -> Tuple[List[str], List[str]]:
        """
        Load the dataset from files, building features and labels
        :return: list of features and labels
        """
        features, labels = self.load_files()
        x = [self._process_text(f, self.max_length) for f in features]
        y = [self._convert_labels(l, self.max_length) for l in labels]
        return x, y

    def load_files(self) -> Tuple[List[str], List[str]]:
        """
        Load features and labels from files
        :return: list of features and labels
        """
        filename, _, ext = self.file_path.rpartition(".")
        features_file = filename + "_nospace." + ext
        labels_file = filename + "_bies." + ext
        features = self.read_dataset(features_file)
        labels = self.read_dataset(labels_file)
        avg_len = sum(len(s) for s in features) // len(features)
        print("Dataset average length:", avg_len)
        self.max_length = avg_len
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


class DatasetLM(Dataset):
    def __init__(
        self,
        file_path: str,
        language_model: str = "bert-base-chinese",
        max_length: int = 200,
    ):
        super().__init__(file_path)
        # self.tokenizer = tr.AutoTokenizer.from_pretrained(language_model)
        self.tokenizer = tr.BertTokenizerFast.from_pretrained(
            language_model  # , tokenize_chinese_chars=True
        )
        self.max_length = max_length
        self.features, self.labels = self.process_data()

    def _process_text(self, text: str, max_length: int) -> Dict[str, Sequence[int]]:
        """
        Preprocess the text according to the language model
        tokenizer
        :param text: text to preprocess
        :return: a dictionaty containg the data for the model
        """
        inputs = self.tokenizer(
            [c for c in text],
            return_token_type_ids=True,
            return_attention_mask=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            is_pretokenized=True,
        )
        return inputs.data

    def _convert_labels(self, bies_line: str, max_length: int) -> Sequence[int]:
        """
        Convert a BIES line to integer based label for the model
        :param bies_line: BIES line in input
        :return: the same line with numerical labels
        """
        converted_line = [self.bies_dict[c] for c in bies_line[: max_length]]
        converted_line = [0] + converted_line + [0]
        if len(converted_line) < max_length:
            converted_line += [0] * (max_length - len(converted_line))
        return converted_line

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


class DatasetLSTM(DatasetLM):
    def __init__(
        self,
        file_path: str,
        word_vectors: gensim.models.word2vec.Word2Vec,
        max_length: int = 200,
    ):
        super().__init__(file_path)
        self.max_length = max_length
        self.word_vectors = word_vectors
        # build the vocab from the w2v model
        self.vocab = self.vocab_from_w2v(self.word_vectors)
        self.features, self.labels = self.process_data()

    @staticmethod
    def _process_text(
        text: str, max_length: int, pad: bool = True
    ) -> Tuple[Sequence[int], Sequence[int]]:
        """
        Preprocess the text by computing unigram and bigram
        :param text: text to preprocess
        :return: a tuple containg the data for the model
        """
        input_unigrams = [c for c in text]
        text_bigrams = DatasetLSTM.compute_bigrams(text)
        input_bigrams = [c for c in text_bigrams]
        # cut to max len
        input_unigrams = input_unigrams[:max_length]
        input_bigrams = input_bigrams[:max_length]
        # pad sequences
        if pad and len(input_unigrams) < max_length:
            input_unigrams += ["<PAD>"] * (max_length - len(input_unigrams))
        if pad and len(input_bigrams) < max_length:
            input_bigrams += ["<PAD>"] * (max_length - len(input_bigrams))
        return input_unigrams, input_bigrams

    def _convert_labels(self, bies_line: str, max_length: int) -> Sequence[int]:
        """
        Convert a BIES line to integer based label for the model
        :param bies_line: BIES line in input
        :return: the same line with numerical labels
        """
        converted_line = [self.bies_dict[c] for c in bies_line[:max_length]]
        if len(converted_line) < max_length:
            converted_line += [0] * (max_length - len(converted_line))
        return converted_line

    @staticmethod
    def vocab_from_w2v(word_vectors: gensim.models.word2vec.Word2Vec) -> Dict[str, int]:
        """
        :param word2vec: trained Gensim Word2Vec model
        :return: a dictionary from token to int
        """
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for index, word in enumerate(word_vectors.wv.index2word):
            vocab[word] = index + 2
        return vocab

    @staticmethod
    def compute_bigrams(line: str) -> List[str]:
        """
        Computes bigrams from the given line
        :param line: line to process
        :return: list of bigrams for the given line
        """
        return DatasetLSTM.pairwise(chain(line, ["</s>"]))

    @staticmethod
    def pairwise(iterable):
        """
        Returns a list of paired items, overlapping, from the original.
        """
        a, b = tee(iterable)
        next(b, None)
        return ["".join(t) for t in zip(a, b)]

    @staticmethod
    def generate_batch(batch, vocab):
        input_unigrams = [DatasetLSTM._encode_sequence(b[0][0], vocab) for b in batch]
        input_bigrams = [DatasetLSTM._encode_sequence(b[0][1], vocab) for b in batch]
        input_unigrams = torch.tensor(input_unigrams)
        input_bigrams = torch.tensor(input_bigrams)
        labels = torch.tensor([b[1] for b in batch])
        return (input_unigrams, input_bigrams), labels

    @staticmethod
    def _encode_sequence(text: str, vocab: Dict):
        return [vocab[ngram] if ngram in vocab else vocab["<UNK>"] for ngram in text]
