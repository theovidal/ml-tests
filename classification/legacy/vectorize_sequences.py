from sklearn.feature_extraction.text import CountVectorizer


def get_vocabulary(train_texts, num_features=20_000):
    kwargs = {
        'lowercase': True,
        'min_df': 2,
        'strip_accents': 'unicode',
        'max_features': num_features
    }

    count = CountVectorizer(**kwargs)

    count.fit(train_texts)

    return list(count.vocabulary_.keys())
