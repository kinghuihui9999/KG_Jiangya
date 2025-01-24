import os

BASE_PATH = os.path.dirname(__file__)

TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, './data/input/train.csv')
TEST_SAMPLE_PATH = os.path.join(BASE_PATH, './data/input/test.csv')

LABEL_PATH = os.path.join(BASE_PATH, './data/input/label.txt')
VOCAB_PATH = os.path.join(BASE_PATH, './data/input/vocab.txt')

BERT_PAD_ID = 0
TEXT_LEN = 70

BATCH_SIZE = 100

BERT_MODEL = os.path.join(BASE_PATH, '../huggingface/bert-base-chinese/')
MODEL_DIR = os.path.join(BASE_PATH, './data/output/models/')

# EMBEDDING_DIM = 768
EMBEDDING_DIM = 300
NUM_FILTERS = 256
NUM_CLASSES = 5
FILTER_SIZES = [2, 3, 4]
VOCAB_SIZE = 784
EPOCH = 70
LR = 1e-5

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))