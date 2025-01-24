import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from config import *  # 确保这里定义了所有需要的配置，如BERT_MODEL, EMBEDDING_DIM等


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,  # 假设EMBEDDING_DIM与BERT模型的隐藏层大小相同
            hidden_size=HIDDEN_SIZE,
            num_layers=1,  # 可以根据需要调整层数
            batch_first=True,
            bidirectional=True,
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=2 * HIDDEN_SIZE,  # 因为BiLSTM的输出是双向的，所以维度是HIDDEN_SIZE的两倍
            num_heads=2,  # 多头注意力中的头数，需要自己设定
            batch_first=True,  # 与其他层保持一致，使用batch_first格式
        )

        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)  # 假设多头注意力的输出维度与BiLSTM相同
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input, mask):
        out = self.bert(input, attention_mask=mask)[0]
        out, _ = self.lstm(out)
        return out

    def _apply_attention(self, lstm_out, mask):
        # 将mask转换为布尔类型，假设非零值表示有效，零值表示填充
        key_padding_mask = mask.bool()  # 转换为布尔型张量
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out, key_padding_mask=~key_padding_mask)
        return self.linear(attn_output)

    def forward(self, input, mask):
        lstm_out = self._get_lstm_feature(input, mask)
        out = self._apply_attention(lstm_out, mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        lstm_out = self._get_lstm_feature(input, mask)
        y_pred = self._apply_attention(lstm_out, mask)
        return -self.crf(y_pred, target, mask, reduction='mean')


if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50))
    mask = torch.ones(100, 50).byte()  # 为了简化假设所有输入都有效

    decoded_sequences = model(input, mask)

    print("解码序列的数量:", len(decoded_sequences))
    if decoded_sequences:
        print("第一个解码序列的长度:", len(decoded_sequences[0]))
