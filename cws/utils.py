import gensim


def convert_w2v_to_binary(input_file, output_file):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(input_file)
    word_vectors.save_word2vec_format(output_file, binary=True)


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


if __name__ == "__main__":
    convert_w2v_to_binary(
        "resources/bigram_unigram300.txt", "resources/bigram_unigram300.bin"
    )

