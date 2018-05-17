import aes.utils as U
import argparse
import keras.backend as K
import logging
import numpy as np
import os
import pickle as pk
from aes.models import Models
import keras.optimizers as opt
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-tr',
    '--train',
    dest='train_path',
    type=str,
    metavar='<str>',
    required=True,
    help='The path to the training set')
parser.add_argument(
    '-tu',
    '--tune',
    dest='dev_path',
    type=str,
    metavar='<str>',
    required=True,
    help='The path to the development set')
parser.add_argument(
    '-ts',
    '--test',
    dest='test_path',
    type=str,
    metavar='<str>',
    required=True,
    help='The path to the test set')
parser.add_argument(
    '-o',
    '--out-dir',
    dest='out_dir',
    type=str,
    metavar='<str>',
    required=True,
    help='The path to the output directory')
parser.add_argument(
    '-p',
    '--prompt',
    dest='prompt_id',
    type=int,
    metavar='<int>',
    required=False,
    help='Promp ID for ASAP dataset. 0 means all prompts.')
parser.add_argument(
    '-m',
    '--model-type',
    dest='model_type',
    type=str,
    metavar='<str>',
    default='gate-matrix',
    help='Model type (gate-positional|gate-matrix|gate-vector|concat|' +
    'char-cnn|char-lstm|char-gru|char-rnn|' +
    'word-cnn|word-lstm|word-gru|word-rnn)' + '(default=gate-matrix)')
parser.add_argument(
    '--emb',
    dest='emb_path',
    type=str,
    metavar='<str>',
    help='The path to the word embeddings file (Word2Vec format)')
parser.add_argument(
    '-v',
    '--vocab-word-size',
    dest='vocab_word_size',
    type=int,
    metavar='<int>',
    default=4000,
    help='Word vocab size (default=4000)')
parser.add_argument(
    '--emb-dim',
    dest='emb_dim',
    type=int,
    metavar='<int>',
    default=50,
    help='Embeddings dimension (default=50)')
parser.add_argument(
    '-b',
    '--batch-size',
    dest='batch_size',
    type=int,
    metavar='<int>',
    default=32,
    help='Batch size (default=32)')
parser.add_argument(
    '-e',
    '--epochs',
    dest='epochs',
    type=int,
    metavar='<int>',
    default=50,
    help='Embeddings dimension (default=50)')
parser.add_argument(
    '-cpw',
    '--char-per-word',
    dest='char_per_word',
    type=int,
    metavar='<int>',
    default=7,
    help='Character per word. (default=7)')
parser.add_argument(
    '-ccnn',
    '--char-cnn-kernel',
    dest='char_cnn_kernel',
    type=int,
    metavar='<int>',
    default=3,
    help='Character CNN kernel size. (default=3)')
parser.add_argument(
    '-cnn',
    '--cnn-kernel',
    dest='cnn_kernel',
    type=int,
    metavar='<int>',
    default=3,
    help='CNN kernel size. (default=3)')
# Optional arguments
parser.add_argument(
    '-cvp',
    '--vocab-char-path',
    dest='vocab_char_path',
    type=str,
    metavar='<str>',
    help='(Optional) The path to the existing char vocab file (*.pkl)')
parser.add_argument(
    '-vp',
    '--vocab-path',
    dest='vocab_word_path',
    type=str,
    metavar='<str>',
    help='(Optional) The path to the existing vocab file (*.pkl)')
# Get all arguments
args = parser.parse_args()
train_path, dev_path, test_path, out_dir, \
    prompt_id, model_type, emb_path, vocab_word_size, \
    emb_dim, batch_size, epochs, \
    char_cnn_kernel, cnn_kernel, \
    char_per_word, vocab_char_path, vocab_word_path = \
    args.train_path, args.dev_path, args.test_path, args.out_dir, \
    args.prompt_id, args.model_type, args.emb_path, args.vocab_word_size, \
    args.emb_dim, args.batch_size, args.epochs, \
    args.char_cnn_kernel, args.cnn_kernel, \
    args.char_per_word, args.vocab_char_path, args.vocab_word_path

if prompt_id == 2 and model_type in ['word-cnn', 'concat']:
    np.random.seed(11)
elif prompt_id == 7 and model_type in ['word-cnn', 'concat']:
    np.random.seed(113)
else:
    np.random.seed(1234)

assert model_type in {
    'gate-positional', 'gate-matrix', 'gate-vector', 'concat', 'char-cnn',
    'char-lstm', 'char-gru', 'char-rnn', 'word-cnn', 'word-lstm', 'word-gru',
    'word-rnn'
}
U.mkdir_p(out_dir + '/preds')
U.set_logger(out_dir)
U.print_args(args)
if 'word' in model_type:
    char_per_word = 0
logger.info('char_per_word: ' + str(char_per_word))

#Remove if statement since no different for all prompt_idt / unique prompt_id
from aes.evaluator import Evaluator
import aes.reader as dataset

# Get data
(train_x_char, train_x, train_y, train_pmt), \
    (dev_x_char, dev_x, dev_y, dev_pmt), \
    (test_x_char, test_x, test_y, test_pmt), \
    vocab_char, vocab_char_size, char_maxlen, \
    vocab_word, vocab_word_size, word_maxlen, \
    num_outputs = \
    dataset.get_data(
         paths=(train_path, dev_path, test_path),
         prompt_id=prompt_id,
         char_per_word=char_per_word,
         vocab_char_path=vocab_char_path,
         vocab_word_size=vocab_word_size,
         vocab_word_path=vocab_word_path)
# Dump vocab
with open(out_dir + '/vocab_word.pkl', 'wb') as vocab_word_file:
    pk.dump(vocab_word, vocab_word_file)
with open(out_dir + '/vocab_char.pkl', 'wb') as vocab_char_file:
    pk.dump(vocab_char, vocab_char_file)
# Pad sequences for mini-batch processing for word level
logger.info('Processing word data')
train_x = sequence.pad_sequences(
    train_x, maxlen=word_maxlen, padding='post', truncating='post')
dev_x = sequence.pad_sequences(
    dev_x, maxlen=word_maxlen, padding='post', truncating='post')
test_x = sequence.pad_sequences(
    test_x, maxlen=word_maxlen, padding='post', truncating='post')
# Pad sequences for mini-batch processing for char level
if 'word' not in model_type:
    logger.info('Processing character data')
    train_x_char = sequence.pad_sequences(
        train_x_char, maxlen=char_maxlen, padding='post', truncating='post')
    train_x_char = np.reshape(train_x_char, (len(train_x_char), -1))
    dev_x_char = sequence.pad_sequences(
        dev_x_char, maxlen=char_maxlen, padding='post', truncating='post')
    dev_x_char = np.reshape(dev_x_char, (len(dev_x_char), -1))
    test_x_char = sequence.pad_sequences(
        test_x_char, maxlen=char_maxlen, padding='post', truncating='post')
    test_x_char = np.reshape(test_x_char, (len(test_x_char), -1))
    char_maxlen = len(test_x_char[0])
# Some statistics
train_y = np.array(train_y, dtype=K.floatx())
dev_y = np.array(dev_y, dtype=K.floatx())
test_y = np.array(test_y, dtype=K.floatx())

train_pmt = np.array(train_pmt, dtype='int32')
dev_pmt = np.array(dev_pmt, dtype='int32')
test_pmt = np.array(test_pmt, dtype='int32')

bincounts, mfs_list = U.bincounts(train_y)

with open('%s/bincounts.txt' % out_dir, 'w') as output_file:
    for bincount in bincounts:
        output_file.write(str(bincount) + '\n')
train_mean = train_y.mean(axis=0)
train_std = train_y.std(axis=0)
dev_mean = dev_y.mean(axis=0)
dev_std = dev_y.std(axis=0)
test_mean = test_y.mean(axis=0)
test_std = test_y.std(axis=0)
logger.info('Statistics:')
logger.info('  train_x shape: ' + str(np.array(train_x).shape))
logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
logger.info('  test_x shape:  ' + str(np.array(test_x).shape))
if 'word' not in model_type:
    logger.info('  train_x_char shape: ' + str(np.array(train_x_char).shape))
    logger.info('  dev_x_char shape:   ' + str(np.array(dev_x_char).shape))
    logger.info('  test_x_char shape:  ' + str(np.array(test_x_char).shape))
logger.info('  train_y shape: ' + str(train_y.shape))
logger.info('  dev_y shape:   ' + str(dev_y.shape))
logger.info('  test_y shape:  ' + str(test_y.shape))
logger.info('  train_y mean: %s, stdev: %s, MFC: %s' %
            (str(train_mean), str(train_std), str(mfs_list)))
# Dev and test sets needs to be in the original scale for evaluation
dev_y_org = dev_y.astype(dataset.get_ref_dtype())
test_y_org = test_y.astype(dataset.get_ref_dtype())
# Convert scores to boundary of [0 1] for training and evaluation
# (loss calculation)
train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
test_y = dataset.get_model_friendly_scores(test_y, test_pmt)
# Building model
models = Models(prompt_id=prompt_id, initial_mean_value=train_y.mean(axis=0))
if model_type == 'gate-positional':
    model = models.create_gate_positional_model(
        char_cnn_kernel=char_cnn_kernel,
        cnn_kernel=cnn_kernel,
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'gate-matrix':
    model = models.create_gate_matrix_model(
        char_cnn_kernel=char_cnn_kernel,
        cnn_kernel=cnn_kernel,
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'gate-vector':
    model = models.create_gate_vector_model(
        char_cnn_kernel=char_cnn_kernel,
        cnn_kernel=cnn_kernel,
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'concat':
    model = models.create_concat_model(
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'char-cnn':
    model = models.create_char_cnn_model(
        emb_dim=emb_dim,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'char-lstm':
    model = models.create_char_lstm_model(
        emb_dim=emb_dim,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'char-gru':
    model = models.create_char_gru_model(
        emb_dim=emb_dim,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'char-rnn':
    model = models.create_char_rnn_model(
        emb_dim=emb_dim,
        word_maxlen=word_maxlen,
        vocab_char_size=vocab_char_size,
        char_maxlen=char_maxlen)
elif model_type == 'word-cnn':
    model = models.create_word_cnn_model(
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen)
elif model_type == 'word-lstm':
    model = models.create_word_lstm_model(
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen)
elif model_type == 'word-gru':
    model = models.create_word_gru_model(
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen)
elif model_type == 'word-rnn':
    model = models.create_word_rnn_model(
        emb_dim=emb_dim,
        emb_path=emb_path,
        vocab_word=vocab_word,
        vocab_word_size=vocab_word_size,
        word_maxlen=word_maxlen)
model.compile(
    loss='mean_squared_error',
    optimizer=opt.RMSprop(
        lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=10, clipvalue=0),
    metrics=['mean_absolute_error'])
logger.info(model.summary())
plot_model(model, to_file=out_dir + '/model.png')  # Plotting model
# Save model architecture
logger.info('Saving model architecture')
with open(out_dir + '/model_arch.json', 'w') as arch:
    arch.write(model.to_json(indent=2))
logger.info('  Done')
# Evaluator
if model_type in ['gate-positional', 'gate-matrix', 'gate-vector', 'concat']:
    evaluator = Evaluator(
        model_type=model_type,
        batch_size=batch_size,
        dataset=dataset,
        prompt_id=prompt_id,
        out_dir=out_dir,
        dev_x=[dev_x_char, dev_x],
        test_x=[test_x_char, test_x],
        dev_y=dev_y,
        test_y=test_y,
        dev_y_org=dev_y_org,
        test_y_org=test_y_org)
elif 'char' in model_type:
    evaluator = Evaluator(
        model_type=model_type,
        batch_size=batch_size,
        dataset=dataset,
        prompt_id=prompt_id,
        out_dir=out_dir,
        dev_x=dev_x_char,
        test_x=test_x_char,
        dev_y=dev_y,
        test_y=test_y,
        dev_y_org=dev_y_org,
        test_y_org=test_y_org)
else:
    evaluator = Evaluator(
        model_type=model_type,
        batch_size=batch_size,
        dataset=dataset,
        prompt_id=prompt_id,
        out_dir=out_dir,
        dev_x=dev_x,
        test_x=test_x,
        dev_y=dev_y,
        test_y=test_y,
        dev_y_org=dev_y_org,
        test_y_org=test_y_org)
logger.info(
    '-------------------------------------------------------------------------'
)
logger.info('Initial Evaluation:')
evaluator.evaluate(model=model, epoch=-1, print_info=True)
total_train_time = 0
total_eval_time = 0
for ii in range(epochs):
    # Training
    t0 = time()
    if model_type in ['gate-positional', 'gate-matrix', 'gate-vector',
                      'concat']:
        train_history = model.fit(
            [train_x_char, train_x],
            train_y,
            batch_size=batch_size,
            epochs=1,
            verbose=0)
    elif 'char' in model_type:
        train_history = model.fit(
            train_x_char, train_y, batch_size=batch_size, epochs=1, verbose=0)
    else:
        train_history = model.fit(
            train_x, train_y, batch_size=batch_size, epochs=1, verbose=0)
    tr_time = time() - t0
    total_train_time += tr_time
    # Evaluate
    t0 = time()
    evaluator.evaluate(model=model, epoch=ii)
    evaluator_time = time() - t0
    total_eval_time += evaluator_time
    # Print information
    train_loss = train_history.history['loss'][0]
    train_metric = train_history.history['mean_absolute_error'][0]
    logger.info('Epoch %d, train: %is, evaluation: %is' % (ii, tr_time,
                                                           evaluator_time))
    logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss,
                                                      train_metric))
    evaluator.print_info()
# Summary of the results
logger.info('Training:   %i seconds in total' % total_train_time)
logger.info('Evaluation: %i seconds in total' % total_eval_time)
evaluator.print_final_info()
