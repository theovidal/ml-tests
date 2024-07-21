import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import utils

np.set_printoptions(precision=3, suppress=True)

def separate_text_and_label(data_path, data_type='train'):
    texts = []
    labels = []
    for category in ['pos', 'neg']:
        full_path = os.path.join(data_path, data_type, category)
        for fname in sorted(os.listdir(full_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(full_path, fname)) as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)

    return texts, labels

def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    train_texts, train_labels = separate_text_and_label(imdb_data_path)
    test_texts, test_labels = separate_text_and_label(imdb_data_path, type='test')

    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def main():



if __name__ == '__main__':
    main()
    plt.show()
