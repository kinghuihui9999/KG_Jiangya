from torch.utils import data
from config import *
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report
import pandas as pd

from transformers import logging
logging.set_verbosity_error()

# class Dataset(data.Dataset):
#     def __init__(self, type='train'):
#         super().__init__()
#         if type == 'train':
#             sample_path = TRAIN_SAMPLE_PATH
#         elif type == 'test':
#             sample_path = TEST_SAMPLE_PATH
#
#         self.lines = pd.read_csv(sample_path)
#         self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
#
#     def __len__(self):
#         return len(self.lines)
#
#     def __getitem__(self, index):
#         text, label = self.lines.loc[index, ['text', 'label_id']]
#         tokened = self.tokenizer(text)
#         input_ids = tokened['input_ids']
#         mask = tokened['attention_mask']
#         if len(input_ids) < TEXT_LEN:
#             pad_len = (TEXT_LEN - len(input_ids))
#             input_ids += [BERT_PAD_ID] * pad_len
#             mask += [0] * pad_len
#         target = int(label)
#         return torch.tensor(input_ids[:TEXT_LEN]), torch.tensor(mask[:TEXT_LEN]), torch.tensor(target)
#
#
# def get_label():
#     text = open(LABEL_PATH,encoding='utf-8').read()
#     id2label = text.split()
#     return id2label, {v: k for k, v in enumerate(id2label)}
#
#
# def evaluate(pred, true, target_names=None, output_dict=False):
#     return classification_report(
#         true,
#         pred,
#         target_names=target_names,
#         labels=range(NUM_CLASSES),
#         output_dict=output_dict,
#         zero_division=0,
#     )
#
# if __name__ == '__main__':
#     dataset = Dataset()
#     loader = data.DataLoader(dataset, batch_size=2)
#     print(next(iter(loader)))





from torch.utils import data
import torch
import pandas as pd

def load_vocab():
    vocab = {}
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split(',')
            vocab[word] = int(idx)
    return vocab

vocab = load_vocab()

class Dataset(data.Dataset):
    def __init__(self, type='train', vocab= vocab, sample_path=''):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH

        self.lines = pd.read_csv(sample_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines.loc[index, ['text', 'label_id']]
        # 将文本转换为字符序列，然后使用vocab进行索引映射
        indexed_text = [self.vocab.get(char, self.vocab['<UNK>']) for char in list(text)]

        # 应用padding
        TEXT_LEN = 100  # 根据你的需求调整
        if len(indexed_text) < TEXT_LEN:
            pad_len = TEXT_LEN - len(indexed_text)
            indexed_text += [self.vocab['<PAD>']] * pad_len

        target = int(label)
        return torch.tensor(indexed_text[:TEXT_LEN], dtype=torch.long), torch.tensor(target)

def get_label():
    text = open(LABEL_PATH,encoding='utf-8').read()
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}

def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        target_names=target_names,
        labels=range(NUM_CLASSES),
        output_dict=output_dict,
        zero_division=0,
    )

if __name__ == '__main__':
    # vocab = load_vocab('./data/input/vocab.txt')  # 确保这是正确的路径
    # 在这里传递vocab
    # dataset = Dataset(type='train', vocab=vocab)#,
    # sample_path='path_to_your_train.csv')  # 传入正确的训练样本路径
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2)
    for data in loader:
        print(data)
        break  # 只打印第一批数据，避免输出过多
