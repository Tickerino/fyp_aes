import logging
import nltk
import numpy as np
import pickle as pk
import re
from keras.preprocessing import sequence

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

asap_ranges = {
    0: (0, 3)
}

def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def convert_to_dataset_friendly_scores(scores_array):
    low, high = asap_ranges[int(0)]
    scores_array = scores_array * (high - low) + low
    assert np.all(scores_array >= low) and np.all(scores_array <= high)
    return scores_array


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_word_path):
    logger.info('Loading vocabulary from: ' + vocab_word_path)
    with open(vocab_word_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def load_vocab_char(vocab_char_path):
    logger.info('Loading vocabulary from: ' + vocab_char_path)
    with open(vocab_char_path, 'rb') as vocab_file:
        vocab_char = pk.load(vocab_file)
    return vocab_char


def process_essay(essay, input_w_shape, input_c_shape):
    vocab = load_vocab('main/model/vocab_word.pkl')
    logger.info("  Vocab size: %i" % (len(vocab)))
    vocab_char = load_vocab_char('main/model/vocab_char.pkl')
    logger.info('  vocab_char size: %i' % (len(vocab_char)))

    content = tokenize(essay)
    indices, char_indices = [], []
    for word in content:
        if is_number(word):
            indices.append(vocab["<num>"])
        elif word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab["<unk>"])
        word = re.sub(r'[^\x00-\x7F]+', '', word)
        char_int = []
        for c in word:
            char_int.append(vocab_char[c] if c in vocab_char
                            else vocab_char['<unk>'])
            if len(char_int) >= 7:
                break
        while len(char_int) < 7:
            char_int.append(vocab_char['<pad>'])
        char_indices.append(char_int)

    data_x_c = np.array([char_indices])
    data_x = np.array([indices])

    data_x_c = sequence.pad_sequences(
        data_x_c, maxlen=input_w_shape[1], padding='post', truncating='post')
    data_x_c = np.reshape(data_x_c, (len(data_x_c), -1))
    data_x = sequence.pad_sequences(
        data_x, maxlen=input_w_shape[1], padding='post', truncating='post')

    return [data_x_c, data_x]
