from config import *
import pandas as pd

import matplotlib.pyplot as plt

def count_text_len():
    text_len = []
    lines = pd.read_csv(TRAIN_SAMPLE_PATH, encoding='utf-8')
    for line in lines['text']:
        text_len.append(len(line))
    plt.hist(text_len)
    plt.show()
    print(max(text_len))

if __name__ == '__main__':
    count_text_len()