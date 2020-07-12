import unicodedata
from typing import List, Sequence


def preprocess_file(filename: str, output_no_space: str, output_bies: str):
    """
    Write an half-width normalized file without spaces and a file in BIES format.
    :param filename: filename in input
    :param output_no_space: filename of the output without spaces
    :param output_bies: filename of the output in BIES format
    """
    dataset = read_dataset(filename)
    dataset = [half_width_char(l) for l in dataset]
    write_dataset(dataset, filename)
    remove_space(dataset, output_no_space)
    compute_bies(filename, output_bies)


def write_dataset(lines: List, filename: str):
    """
    Writes a list of string in a file.
    :param filename: path where to save the file.
    :param lines: list of strings to serilize.
    :return:
    """
    with open(filename, "w", encoding="utf8") as file:
        file.writelines(line + "\n" for line in lines)


def compute_bies(input_file: Sequence[str], output_file: str):
    """
    Produce a label file in BIES format from the given input file.
    :param input_file: input filename to read
    :param output_file: output filename to write
    """
    out_ctx = open(output_file, mode="w", encoding="utf-8")
    with out_ctx as out_file:
        out_file.writelines(_bies_line(l.strip()) + "\n" for l in input_file)


def _bies_line(line: str) -> str:
    """
    :param line: line to convert in BIES tagging
    :return: BIES tagging converted line
    """
    return "".join(_bies_word(half_width_char(w)) for w in line.split())


def _bies_word(word: str) -> str:
    """
    :param word: word to convert in BIES tagging
    :return: BIES tagging converted word
    """
    # if it's a char-word, tag it with S
    if len(word) == 1:
        return "S"
    # else, B -> first char, I -> mid chars, E -> last char
    else:
        return "B" + "".join("I" for _ in word[1:-1]) + "E"


def remove_space(input_file: Sequence[str], output_file: str):
    """
    Removes spaces in training file.
    :param input_file: path of the input file
    :param output_file: path of the output file
    :return: input file without spaces
    """
    out_ctx = open(output_file, mode="w", encoding="utf-8")
    with out_ctx as out_file:
        out_file.writelines(half_width_char(l.replace(" ", "")) for l in input_file)


def half_width_char(line: str) -> str:
    """
    Replaces full-width characters by half-width ones.
    :param line: string to process
    :return: the input string with half-width character.
    """
    return unicodedata.normalize("NFKC", line)


def read_dataset(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def timer(start: float, end: float):
    """
    Timer function. Compute execution time from strart to end (end - start)
    :param start: start time
    :param end: end time
    :return: end - start
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
