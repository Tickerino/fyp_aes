import logging
import nltk
import numpy as np
import pickle as pk
import re

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'


asap_ranges = {
    0: (0, 3),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 3),
    6: (0, 3),
    7: (0, 30),
    8: (0, 60)
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


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]


def get_model_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = (scores_array - low) / (high - low)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = (scores_array - low) / (high - low)
    assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    return scores_array


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
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


def create_vocab(file_path, prompt_id, vocab_word_size):
    logger.info('Creating vocabulary from: ' + file_path)
    total_words, unique_words = 0, 0
    word_freqs = {}
    with open(file_path, 'r') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_set = int(tokens[1])
            content = tokens[2].decode('latin1').strip()
            if essay_set == prompt_id or prompt_id <= 0:
                content = content.lower()
                content = tokenize(content)
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[word] = 1
                    total_words += 1
    logger.info('  %i total words, %i unique words' % (total_words,
                                                       unique_words))
    import operator
    sorted_word_freqs = sorted(
        word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_word_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def create_vocab_char(file_path, prompt_id):
    logger.info('Creating char vocabulary from: ' + file_path)
    combined_essay = ''
    vocab_char = {'<pad>': 0, '<unk>': 1}
    with open(file_path, 'r') as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_set = int(tokens[1])
            content = tokens[2].decode('latin1').strip()
            if essay_set == prompt_id or prompt_id <= 0:
                content = content.lower()
                combined_essay = combined_essay + content + '\n'
        combined_essay = re.sub(r'[^\x00-\x7F]+', ' ', combined_essay)
        combined_essay = re.sub(r'[\n ]+', '', combined_essay)
        import collections
        chars = collections.Counter(combined_essay)
        chars = {k: v for k, v in chars.iteritems() if v > 1}
        vocab_char.update(dict((c, i + 2) for i, c in enumerate(chars)))
    return vocab_char

#New created normalizing switch
# def normalize_score(argument, score):
#     switcher = {
#         1: float((score-2)*10/10),
#         2: float((score)*10/5),
#         3: float(score*10/3),
#         4: float(score*10/3),
#         5: float(score*10/4),
#         6: float(score*10/4),
#         7: float(score*10/30),
#         7: float(score*10/60),
#     }
#     return switcher.get(argument,0)

def read_dataset(file_path,
                 prompt_id,
                 char_per_word,
                 vocab_char,
                 vocab_word,
                 score_index=6):
    logger.info('Reading dataset from: ' + file_path)
    data_char_x, data_x, data_y, prompt_ids = [], [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_word = -1
    maxlen_char = 0
    with open(file_path) as input_file:
        input_file.next()
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_set = int(tokens[1])
            content = tokens[2].decode('latin1').strip()
            score = float(tokens[score_index])
            #score = float(normalize_score(essay_set,score))#I added this line
            if essay_set == prompt_id or prompt_id <= 0:
                content = content.lower()
                content = tokenize(content)
                char_indices, word_indices = [], []
                for word in content:
                    if is_number(word):
                        word_indices.append(vocab_word['<num>'])
                        num_hit += 1
                    elif word in vocab_word:
                        word_indices.append(vocab_word[word])
                    else:
                        word_indices.append(vocab_word['<unk>'])
                        unk_hit += 1
                    total += 1
                    if char_per_word > 0:
                        word = re.sub(r'[^\x00-\x7F]+', '', word)
                        char_int = []
                        for c in word:
                            char_int.append(vocab_char[c] if c in vocab_char
                                            else vocab_char['<unk>'])
                            if len(char_int) >= char_per_word:
                                break
                        while len(char_int) < char_per_word:
                            char_int.append(vocab_char['<pad>'])
                        char_indices.append(char_int)
                        if len(char_indices) > maxlen_char:
                            maxlen_char = len(char_indices)
                data_x.append(word_indices)
                if maxlen_word < len(word_indices):
                    maxlen_word = len(word_indices)
                if char_per_word > 0:
                    data_char_x.append(char_indices)
                data_y.append(score)
                prompt_ids.append(essay_set)
    logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' %
                (100 * num_hit / total, 100 * unk_hit / total))
    return data_char_x, data_x, data_y, prompt_ids, maxlen_char, maxlen_word


def get_data(paths,
             prompt_id,
             char_per_word,
             vocab_char_path,
             vocab_word_size,
             vocab_word_path,
             score_index=6):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]
    if not vocab_word_path:
        vocab_word = create_vocab(train_path, prompt_id, vocab_word_size)
        if len(vocab_word) < vocab_word_size:
            logger.warning('Vocabualry has only %i words (less than %i)' %
                           (len(vocab_word), vocab_word_size))
        else:
            assert vocab_word_size == 0 or len(vocab_word) == vocab_word_size
    else:
        vocab_word = load_vocab(vocab_word_path)
        if len(vocab_word) != vocab_word_size:
            logger.warning(
                'Vocabualry has %i words which is different from given: %i' %
                (len(vocab_word), vocab_word_size))
    logger.info('  Vocab size: %i' % (len(vocab_word)))
    if char_per_word > 0:
        vocab_char = create_vocab_char(
            train_path, prompt_id) if not vocab_char_path else load_vocab_char(
                vocab_char_path)
        logger.info('  vocab_char size: %i' % (len(vocab_char)))
    else:
        vocab_char = None
    train_x_char, train_x, train_y, train_prompts, \
        train_maxlen_char, train_maxlen_word = \
        read_dataset(
            file_path=train_path,
            prompt_id=prompt_id,
            char_per_word=char_per_word,
            vocab_char=vocab_char,
            vocab_word=vocab_word)
    dev_x_char, dev_x, dev_y, dev_prompts, \
        dev_maxlen_char, dev_maxlen_word = \
        read_dataset(
            file_path=dev_path,
            prompt_id=prompt_id,
            char_per_word=char_per_word,
            vocab_char=vocab_char,
            vocab_word=vocab_word)
    test_x_char, test_x, test_y, test_prompts, \
        test_maxlen_char, test_maxlen_word = \
        read_dataset(
            file_path=test_path,
            prompt_id=prompt_id,
            char_per_word=char_per_word,
            vocab_char=vocab_char,
            vocab_word=vocab_word)
    vocab_word_size = 0 if vocab_word is None else len(vocab_word)
    vocab_char_size = 0 if vocab_char is None else len(vocab_char)
    char_maxlen = max(train_maxlen_char, dev_maxlen_char, test_maxlen_char)
    word_maxlen = max(train_maxlen_word, dev_maxlen_word, test_maxlen_word)
    return ((train_x_char, train_x, train_y,
             train_prompts), (dev_x_char, dev_x, dev_y, dev_prompts),
            (test_x_char, test_x, test_y,
             test_prompts), vocab_char, vocab_char_size, char_maxlen,
            vocab_word, vocab_word_size, word_maxlen, 1)
