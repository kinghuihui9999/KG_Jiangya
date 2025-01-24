import torch.nn as nn
from config import *
from torchcrf import CRF
import torch
from transformers import BertModel

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input, mask):
        # out = self.embed(input)
        out = self.bert(input, mask)[0]
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input, mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input, mask)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')



# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(BERT_MODEL)
#         # 由于移除了BiLSTM层，BERT输出的隐藏状态直接连接到线性层，BERT模型的隐藏状态大小将作为线性层的输入维度
#         self.linear = nn.Linear(self.bert.config.hidden_size, TARGET_SIZE)
#         self.crf = CRF(TARGET_SIZE, batch_first=True)
#
#     def _get_bert_features(self, input, mask):
#         # 获取BERT的输出，第一个输出是最后一层的隐藏状态
#         out = self.bert(input, attention_mask=mask).last_hidden_state
#         # 将BERT的输出通过线性层转换为目标大小
#         out = self.linear(out)
#         return out
#
#     def forward(self, input, mask):
#         out = self._get_bert_features(input, mask)
#         # 使用CRF层进行解码
#         return self.crf.decode(out, mask)
#
#     def loss_fn(self, input, target, mask):
#         y_pred = self._get_bert_features(input, mask)
#         # 计算CRF层的损失
#         return -self.crf(y_pred, target, mask, reduction='mean')
#
#
#
# if __name__ == '__main__':
#     model = Model()
#     input = torch.randint(0, 3000, (100, 50))
#     mask = torch.ones(100, 50).byte()  # 为了简化假设所有输入都有效
#
#     # model(input, mask) 的输出将是解码序列的列表
#     decoded_sequences = model(input, mask)
#
#     # 如果你想打印解码序列的数量和示例解码序列的长度，可以这样做：
#     print("解码序列的数量:", len(decoded_sequences))
#     if decoded_sequences:  # 检查列表是否不为空
#         print("第一个解码序列的长度:", len(decoded_sequences[0]))
