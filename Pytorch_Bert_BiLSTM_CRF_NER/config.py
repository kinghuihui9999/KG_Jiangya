import os

BASE_PATH = os.path.dirname(__file__)


TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, './input/data/train.txt')
TEST_SAMPLE_PATH = os.path.join(BASE_PATH, './input/data/test.txt')
DEV_SAMPLE_PATH = os.path.join(BASE_PATH, './input/data/dev.txt')

LABEL_PATH = os.path.join(BASE_PATH, './output/label.txt')

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

BATCH_SIZE = 20

VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 11
LR = 1e-5
EPOCH = 100

MODEL_DIR = os.path.join(BASE_PATH, './output/model/')

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# bert改造
BERT_MODEL = os.path.join(BASE_PATH, '../huggingface/bert-base-chinese')
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512