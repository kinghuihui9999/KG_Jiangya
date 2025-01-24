import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *
from transformers import BertModel

from transformers import logging
logging.set_verbosity_error()

# class TextCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(BERT_MODEL)
#         for name ,param in self.bert.named_parameters():
#             param.requires_grad = False
#         self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM)) for i in FILTER_SIZES])
#         self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)
#
#     def conv_and_pool(self, conv, input):
#         out = conv(input)
#         out = F.relu(out)
#         return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze(-1).squeeze(-1)
#
#     def forward(self, input, mask):
#         out = self.bert(input, mask)[0].unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
#         return torch.sigmoid(self.linear(out))
#
#
# if __name__ == '__main__':
#     model = TextCNN()
#     input = torch.randint(0, 3000, (2, TEXT_LEN))
#     mask = torch.ones_like(input)
#     print(model(input, mask))
#

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from config import NUM_FILTERS, FILTER_SIZES, EMBEDDING_DIM, NUM_CLASSES
VOCAB_SIZE = 784
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 使用自定义嵌入层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # 应用ReLU激活函数
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 最大池化
        return x

    def forward(self, x):
        x = self.embedding(x)  # 将输入索引转换为嵌入向量
        x = x.unsqueeze(1)  # 增加一个维度以适配Conv2d
        x = [self.conv_and_pool(x, conv) for conv in self.convs]  # 应用所有的卷积和池化层
        x = torch.cat(x, 1)  # 在过滤器维度上拼接
        x = self.dropout(x)  # 应用dropout
        logits = self.fc(x)  # 最终的线性层
        return logits

# 假设的配置值，你需要根据实际情况进行调整
vocab_size = VOCAB_SIZE  # 词汇表大小
embedding_dim = 300  # 嵌入向量维度
num_filters = 100  # 卷积层过滤器数量
filter_sizes = [2, 3, 4]  # 卷积核尺寸
num_classes = NUM_CLASSES  # 类别数量

# 实例化模型
model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes)

