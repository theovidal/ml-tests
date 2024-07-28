from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np


NUM_FEATURES = 20_000
NGRAM_RANGE = (1, 2)


def vectorize_ngrams(train_texts, train_labels, validation_texts):
    kwargs = {
        # For more security, we allow 1-grams to be extracted (if the sentence only contains one word for instance)
        'ngram_range': NGRAM_RANGE,
        # In case accents are present (for instance, if the comments are from other languages than English)
        # Here we know our data is in English, but this parameter could be useful in the future if we want to generalize
        # ou rmodel
        'strip_accents': 'unicode',
        # If other characters than unicode are detected, just replace them ; the essential is having a good vocabulary
        # without repetitions
        'decode_error': 'replace',
        # 'max_features': 20_000, # Will be treated by a feature selection process later

        # Integer are simpler to compute, and we don't need floats for the moment (we are only processing the data here,
        # not interacting with the coefficients)
        'dtype': np.float64,
        # 'norm': None,

        # Remove tokens that shows only once (only keep those which count is more than 2)
        'min_df': 2
    }

    # Step 1 - Feature extraction : we want to extract n-gram frequencies from our texts
    vectorizer = TfidfVectorizer(**kwargs)

    # This operation consists of:
    # - Fitting: read texts, learn the vocabulary and calculate frequencies
    # - Transforming: parse this data into a matrix
    # The returned value is a matrix of shape (n_samples, n_features) : we can get the number of extracted features
    # right from it. It will then be used for our MLP model
    x_train = vectorizer.fit_transform(train_texts)
    # Here we don't need to learn the vocabulary, this was done at the precedent line
    # (and it could cause problems if some words are different)
    x_validation = vectorizer.transform(validation_texts)

    # Step 2 - Feature selection : we only consider the features that are relevant for our purpose We choose the
    # classification function as we face a classification problem

    # Note the difference between the vectorizer - to return a vector out of another object - and the transformer

    # BE CAREFUL: if the dataset has less than 20k
    # features, we need a fallback -> the number of features already present
    selector = SelectKBest(score_func=f_classif, k=min(NUM_FEATURES, x_train.shape[1]))
    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype(np.float64)
    x_validation = selector.transform(x_validation).astype(np.float64)

    return x_train, x_validation


