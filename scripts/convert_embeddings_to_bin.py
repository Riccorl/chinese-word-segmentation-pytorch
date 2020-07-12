import gensim


def convert_w2v_to_binary(input_file, output_file):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(input_file)
    word_vectors.save_word2vec_format(output_file, binary=True)


if __name__ == "__main__":
    convert_w2v_to_binary(
        "resources/bigram_unigram300.txt", "resources/bigram_unigram300.bin"
    )
