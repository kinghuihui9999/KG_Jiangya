import pandas as pd
from collections import Counter


# 步骤 1: 加载数据
def load_data(data_path):
    return pd.read_csv(data_path)


# 步骤 2: 构建词汇表
def build_vocab(texts, vocab_size=3000):
    counter = Counter()
    for text in texts:
        counter.update(list(text))

    most_common = counter.most_common(vocab_size - 2)
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1

    return word2idx


# 步骤 3: 保存词汇表到文件
def save_vocab(vocab, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, idx in vocab.items():
            f.write(f"{word},{idx}\n")


# 主程序
if __name__ == "__main__":
    data_path = './data/input/train.csv'  # 你的数据文件路径
    vocab_file_path = './data/input/vocab.txt'  # 词汇表输出文件路径

    # 加载数据
    data = load_data(data_path)

    # 构建词汇表
    vocab_size = 3000  # 你可以根据需要调整词汇表大小
    vocab = build_vocab(data['text'], vocab_size)

    # 保存词汇表到文件
    save_vocab(vocab, vocab_file_path)

    print(f"Vocabulary saved to {vocab_file_path}")
