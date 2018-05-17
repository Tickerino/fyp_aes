import keras.backend as K
import logging
import numpy as np
from aes.layers import MeanOverTime
from keras.layers import Activation, Dense, Dropout, Input
from keras.layers.embeddings import Embedding
from keras.models import Model

logger = logging.getLogger(__name__)


class Models:
    def __init__(self, prompt_id, initial_mean_value):
        self.prompt_id = prompt_id
        self.initial_mean_value = initial_mean_value
        self.dropout = 0.5
        self.recurrent_dropout = 0.1
        if self.initial_mean_value.ndim == 0:
            self.initial_mean_value = np.expand_dims(
                self.initial_mean_value, axis=1)
        self.num_outputs = len(self.initial_mean_value)
        self.bias = (np.log(self.initial_mean_value) -
                     np.log(1 - self.initial_mean_value)).astype(K.floatx())

    def create_gate_positional_model(self, char_cnn_kernel, cnn_kernel,
                                     emb_dim, emb_path, vocab_word,
                                     vocab_word_size, word_maxlen,
                                     vocab_char_size, char_maxlen):
        from aes.layers import Conv1DMask, GatePositional, MaxPooling1DMask
        logger.info('Building gate positional model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        char_cnn = Conv1DMask(
            filters=emb_dim,
            kernel_size=3,
            padding='same')(char_emb)
        char_input = MaxPooling1DMask(
            pool_size=char_maxlen / word_maxlen, padding='same')(char_cnn)
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_input = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        gate = GatePositional()([char_input, word_input])
        final_input = Dense(50)(gate)
        cnn = Conv1DMask(
            filters=emb_dim,
            kernel_size=3,
            padding='same')(final_input)
        dropped = Dropout(0.5)(cnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=[input_char, input_word], outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_gate_matrix_model(self, char_cnn_kernel, cnn_kernel, emb_dim,
                                 emb_path, vocab_word, vocab_word_size,
                                 word_maxlen, vocab_char_size, char_maxlen):
        from aes.layers import Conv1DMask, GateMatrix, MaxPooling1DMask
        logger.info('Building gate matrix model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        char_cnn = Conv1DMask(
            filters=emb_dim,
            kernel_size=char_cnn_kernel,
            padding='same')(char_emb)
        char_input = MaxPooling1DMask(
            pool_size=char_maxlen / word_maxlen, padding='same')(char_cnn)
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_input = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        gate = GateMatrix()([char_input, word_input])
        final_input = Dense(50)(gate)
        cnn = Conv1DMask(
            filters=emb_dim,
            kernel_size=cnn_kernel,
            padding='same')(final_input)
        dropped = Dropout(0.5)(cnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=[input_char, input_word], outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_gate_vector_model(self, char_cnn_kernel, cnn_kernel, emb_dim,
                                 emb_path, vocab_word, vocab_word_size,
                                 word_maxlen, vocab_char_size, char_maxlen):
        from aes.layers import Conv1DMask, GateVector, MaxPooling1DMask
        logger.info('Building gate vector model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        char_cnn = Conv1DMask(
            filters=emb_dim,
            kernel_size=char_cnn_kernel,
            padding='same')(char_emb)
        char_input = MaxPooling1DMask(
            pool_size=char_maxlen / word_maxlen, padding='same')(char_cnn)
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_input = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        gate = GateVector()([char_input, word_input])
        final_input = Dense(50)(gate)
        cnn = Conv1DMask(
            filters=emb_dim,
            kernel_size=cnn_kernel,
            padding='same')(final_input)
        dropped = Dropout(0.5)(cnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=[input_char, input_word], outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_concat_model(self, emb_dim, emb_path, vocab_word,
                            vocab_word_size, word_maxlen, vocab_char_size,
                            char_maxlen):
        from aes.layers import Conv1DMask, MaxPooling1DMask
        from keras.layers import concatenate
        logger.info('Building concatenation model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        char_cnn = Conv1DMask(
            filters=emb_dim, kernel_size=3, padding='same')(char_emb)
        char_input = MaxPooling1DMask(
            pool_size=char_maxlen / word_maxlen, padding='same')(char_cnn)
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_input = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        merged = concatenate([char_input, word_input], axis=1)
        merged_dropped = Dropout(0.5)(merged)
        final_input = Dense(50)(merged_dropped)
        cnn = Conv1DMask(
            filters=emb_dim, kernel_size=3, padding='same')(final_input)
        dropped = Dropout(0.5)(cnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=[input_char, input_word], outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_char_cnn_model(self, emb_dim, word_maxlen, vocab_char_size,
                              char_maxlen):
        from aes.layers import Conv1DMask
        logger.info('Building character CNN model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        cnn = Conv1DMask(
            filters=emb_dim, kernel_size=3, padding='same')(char_emb)
        dropped = Dropout(0.5)(cnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_char, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        logger.info('  Done')
        return model

    def create_char_lstm_model(self, emb_dim, word_maxlen, vocab_char_size,
                               char_maxlen):
        from keras.layers import LSTM
        logger.info('Building character LSTM model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        lstm = LSTM(
            300,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)(char_emb)
        dropped = Dropout(0.5)(lstm)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_char, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        logger.info('  Done')
        return model

    def create_char_gru_model(self, emb_dim, word_maxlen, vocab_char_size,
                              char_maxlen):
        from keras.layers import GRU
        logger.info('Building character GRU model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        gru = GRU(
            300,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)(char_emb)
        dropped = Dropout(0.5)(gru)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_char, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        logger.info('  Done')
        return model

    def create_char_rnn_model(self, emb_dim, word_maxlen, vocab_char_size,
                              char_maxlen):
        from keras.layers import SimpleRNN
        logger.info('Building character RNN model')
        input_char = Input(shape=(char_maxlen, ), name='input_char')
        char_emb = Embedding(
            vocab_char_size, emb_dim, mask_zero=True)(input_char)
        rnn = SimpleRNN(
            300,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)(char_emb)
        dropped = Dropout(0.5)(rnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_char, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        logger.info('  Done')
        return model

    def create_word_cnn_model(self, emb_dim, emb_path, vocab_word,
                              vocab_word_size, word_maxlen):
        from aes.layers import Conv1DMask
        logger.info('Building word CNN model')
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_emb = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        cnn = Conv1DMask(
            filters=emb_dim, kernel_size=3, padding='same')(word_emb)
        dropped = Dropout(0.5)(cnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_word, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_word_lstm_model(self, emb_dim, emb_path, vocab_word,
                               vocab_word_size, word_maxlen):
        from keras.layers import LSTM
        logger.info('Building word LSTM model')
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_emb = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        lstm = LSTM(
            300,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)(word_emb)
        dropped = Dropout(0.5)(lstm)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_word, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_word_gru_model(self, emb_dim, emb_path, vocab_word,
                              vocab_word_size, word_maxlen):
        from keras.layers import GRU
        logger.info('Building word GRU model')
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_emb = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        gru = GRU(
            300,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)(word_emb)
        dropped = Dropout(0.5)(gru)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_word, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model

    def create_word_rnn_model(self, emb_dim, emb_path, vocab_word,
                              vocab_word_size, word_maxlen):
        from keras.layers import SimpleRNN
        logger.info('Building word SimpleRNN model')
        input_word = Input(shape=(word_maxlen, ), name='input_word')
        word_emb = Embedding(
            vocab_word_size, emb_dim, mask_zero=True,
            name='word_emb')(input_word)
        rnn = SimpleRNN(
            300,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout)(word_emb)
        dropped = Dropout(0.5)(rnn)
        mot = MeanOverTime(mask_zero=True)(dropped)
        densed = Dense(self.num_outputs, name='dense')(mot)
        output = Activation('sigmoid')(densed)
        model = Model(inputs=input_word, outputs=output)
        model.get_layer('dense').bias.set_value(self.bias)
        if emb_path:
            from emb_reader import EmbReader as EmbReader
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=emb_dim)
            model.get_layer('word_emb').embeddings.set_value(
                emb_reader.get_emb_matrix_given_vocab(
                    vocab_word,
                    model.get_layer('word_emb').embeddings.get_value()))
        logger.info('  Done')
        return model
