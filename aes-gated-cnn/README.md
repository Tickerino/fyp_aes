# Gated Word Character Automated Essay Scoring
A convolutional architecture that incorporates character level information and
models the relationship between word and characters using a gating mechanism.

## Running the model
### Set Up
- Python2
- Install Theano
- Install Keras >= 2.0.4 (with Theano backend)
- Download [training_set_rel3.tsv](https://www.kaggle.com/c/asap-aes/data) and
- Put it in the data directory
- Prepare data by running process.py on training_set_rel3.tsv in data directory
- Run train.py

### Options
You can see the list of available options by running:
```bash
python train.py -h
```
### Example
The following command trains the gate matrix model for prompt 1 in the ASAP
dataset, using the training and development data from fold 0 and evaluates it.

```bash
KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32" \
python2 train.py \
-tr data/train_150/train.tsv \
-tu data/train_150/dev.tsv \
-ts data/train_150/test.tsv \
-o output_150 \
-p 0 \
-e 5 \
-m gate-positional \
--emb embeddings.w2v.txt
```
