import os
import random
import numpy as np


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

    # We should always verify if the same classes are in the train and test labels
    unexpected_labels = [v for v in test_labels if v not in train_labels]
    if len(unexpected_labels):
        raise ValueError(f"Unexpected labels in the test set: {unexpected_labels}. Make sure the labels are the same "
                         f"in the training and testing sets")

    return train_texts, np.array(train_labels), train_classes, test_texts, np.array(test_labels), test_classes

