# Based on https://developers.google.com/machine-learning/guides/text-classification

from collections import Counter

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time

import utils

np.set_printoptions(precision=3, suppress=True)


def separate_text_and_label(data_path, data_type='train'):
    """
    Read all the data and parse it (remove HTML and lowercase)

    :param data_path:
    :param data_type:
    :return:
    """
    texts = []
    labels = []
    classes = {}
    for category in ['pos', 'neg']:
        classes[category] = []
        full_path = os.path.join(data_path, data_type, category)
        for filename in sorted(os.listdir(full_path)):
            if filename.endswith('.txt'):
                with open(os.path.join(full_path, filename)) as file:
                    text = file.read().lower().replace('<br />', '')
                    texts.append(text)
                    classes[category].append(text)
                labels.append(0 if category == 'neg' else 1)

    return texts, labels, classes


def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    train_texts, train_labels, train_classes = separate_text_and_label(data_path)
    test_texts, test_labels, test_classes = separate_text_and_label(data_path, data_type='test')

    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)  # Be sure to have exactly the same order in the other array
    random.shuffle(train_labels)

    return train_texts, np.array(train_labels), train_classes, test_texts, np.array(test_labels), test_classes


def get_words(text):
    """
    Get all the words in text
    :param text: the text to extract words from
    :return: list of words
    """
    return text.split()


def number_of_words_from_text(text):
    """
    Get the number of words in text
    :param text: the text to extract words from
    :return: the number of words
    """
    return len(get_words(text))


def flatten_comprehension(list_of_lists):
    """
    Flatten a list of lists
    :param list_of_lists: a list of lists
    :return: a flattened list of lists
    """
    return [item for row in list_of_lists for item in row]


def plot_word_distribution(texts, category):
    """
    Plot the distribution of words and length across samples of text
    :param texts: a list of texts
    :param category: the category of text
    """
    split = list(map(get_words, texts))

    all_words = np.array(flatten_comprehension(split))
    # Count the frequency of each word
    word_counts = Counter(all_words)
    # Extract words and their frequencies
    words, frequencies = zip(*word_counts.most_common(30))

    plt.figure(figsize=(12, 8))  # Increase figure size if needed

    plt.subplot(1, 2, 1)
    plt.bar(words, frequencies)
    plt.title(f'Frequency distribution of words for {category} samples')
    plt.xlabel('Words')
    plt.ylabel('Frequency in all texts')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit labels
    plt.xlim(0, 40)

    nb_words_distribution = np.fromiter(map(lambda wds: len(wds), split), dtype=int)

    plt.subplot(1, 2, 2)
    plt.hist(nb_words_distribution, 40)
    plt.xlabel('Number of words')
    plt.ylabel('Number of samples')
    plt.title(f'Distribution of number of words for {category} samples')


def analyse_dataset(texts, labels, classes):
    """
    Analyse the dataset using these metrics:
    - Number of samples,
    - Number of classes,
    - Number of samples per class,
    - Average number of words per sample,
    - Distribution of words per category and globally,
    - Distribution of number of words per category and globally.

    :param texts: a list of texts
    :param labels: a numpy array classifying each text to a category
    :param classes: a dict of lists of texts for each category
    """
    nb_samples = len(texts)
    classes_list, class_indexes, nb_samples_per_class = np.unique(labels, return_index=True, return_counts=True)
    nb_classes = len(classes_list)

    # np.vectorize maps every element of the array with a specific function
    word_counts = np.vectorize(number_of_words_from_text)(texts)
    # Thanks to this trick we just have to calculate a mean of all the values
    avg_nb_words = np.mean(word_counts)

    print(f'Number of samples: {nb_samples}\nNumber of classes_list: {nb_classes}\nNumber of samples per class:')
    for i in range(len(classes_list)):
        print(f'- {classes_list[i]}: {nb_samples_per_class[i]}')

    print(f'Average number of words per sample: {avg_nb_words}')

    for category in classes.keys():
        plot_word_distribution(classes[category], category)
    plot_word_distribution(texts, 'all')


def main():
    train_texts, train_labels, train_classes, test_texts, test_labels, test_classes = (
        load_imdb_sentiment_analysis_dataset('aclImdb', int(time.time())))
    analyse_dataset(train_texts, train_labels, train_classes)


if __name__ == '__main__':
    main()
    plt.show()
