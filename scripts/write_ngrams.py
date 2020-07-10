import argparse
from typing import List
from itertools import chain, tee


def compute_bigrams(line: str) -> List[str]:
    """
    Computes bigrams from the given line
    :param line: line to process
    :return: list of bigrams for the given line
    """
    return pairwise(chain(line, ["</s>"]))


def pairwise(iterable):
    """
    Returns a list of paired items, overlapping, from the original.
    """
    a, b = tee(iterable)
    next(b, None)
    return ["".join(t) for t in zip(a, b)]


def read_dataset(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def write_dataset(lines: List, filename: str):
    """
    Writes a list of string in a file.
    :param filename: path where to save the file.
    :param lines: list of strings to serilize.
    :return:
    """
    with open(filename, "w", encoding="utf8") as file:
        file.writelines(line + "\n" for line in lines)


def main(input_file, output_file):
    input_f = read_dataset(input_file)
    input_f = [" ".join(compute_bigrams(l)) for l in input_f]
    write_dataset(input_f, output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="input dataset 1", dest="input_one")
    parser.add_argument(help="input dataset 2", dest="input_two")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_one, args.input_two)
