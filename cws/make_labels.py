from argparse import ArgumentParser
from pathlib import Path
from utils import preprocess_file, half_width_char


def main(folder_path):
    input_dir = Path(folder_path)
    for input_file in input_dir.glob("*.utf8"):
        filename = str(input_file.stem)
        extension = str(input_file.suffix)
        nospace_file = str(input_file.parent) + "/" + filename + "_nospace" + extension
        bies_file = str(input_file.parent) + "/" + filename + "_bies" + extension
        preprocess_file(str(input_file), nospace_file, bies_file)


def half(input_file, output_file):
    with open(input_file) as in_file, open(output_file, "w") as out_file:
        for line in input_file:
            line = line.strip()
            line = half_width_char(line)
            out_file.write(line + "\n")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_folder", help="The path of the folder with input files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_folder)
