import codecs
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbReader:
    def __init__(self, emb_path, emb_dim=None):
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings_index = {}
        self.vocab_size = 0
        self.emb_dim = -1
        if 'glove' not in emb_path:
            with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
                for line in emb_file:
                    tokens = line.split()
                    word = tokens[0]
                    vec = tokens[1].split(',')
                    vec = [float(x) for x in vec]
                    if self.emb_dim == -1:
                        self.emb_dim = len(vec)
                        assert self.emb_dim == emb_dim, \
                            'Dimension does not match.'
                    else:
                        assert len(
                            vec) == self.emb_dim, 'Dimension does not match.'
                    self.embeddings_index[word] = vec
                    self.vocab_size += 1
        else:
            with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
                for line in emb_file:
                    values = line.split()
                    word = values[0]
                    vec = np.asarray(values[1:], dtype='float32')
                    if self.emb_dim == -1:
                        self.emb_dim = len(vec)
                        assert self.emb_dim == emb_dim, \
                            'Dimension does not match.'
                    else:
                        assert len(
                            vec) == self.emb_dim, 'Dimension does not match.'
                    self.embeddings_index[word] = vec
                    self.vocab_size += 1
        logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size,
                                                         self.emb_dim))

    def get_emb_given_word(self, word):
        try:
            return self.embeddings_index[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, embeddings):
        counter = 0.
        for word, index in vocab.iteritems():
            try:
                embeddings[index] = self.embeddings_index[word]
                counter += 1
            except KeyError:
                pass
        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' %
                    (counter, len(vocab), 100 * counter / len(vocab)))
        return embeddings

    def get_emb_dim(self):
        return self.emb_dim
