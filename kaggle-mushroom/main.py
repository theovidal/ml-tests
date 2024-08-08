import os
import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from data import open_and_tidy_data, analyse_dataset, parse_data_for_model
from model import create_model, compile_model, train_model
import params


def load_and_parse_data(file='train.csv', train_proportion=0.8):
    dataset = open_and_tidy_data(file)

    if input('Do you want to analyze the training dataset? (y/N): ') == 'y':
        # We analyse both dataset to see if the statistics are quite the same between them, i.e. if our model can be
        # generalized afterward
        analyse_dataset(dataset, name="train")

    dataset_parsed = parse_data_for_model(dataset)


    # Here we only separate our dataset into training and test, as the validation dataset is
    # automatically extracted by the model
    # We also extract all the labels by calling the "pop" method
    train_features = dataset_parsed.sample(frac=train_proportion)
    test_features = dataset_parsed.drop(train_features.index)

    train_labels = train_features.pop('class')
    test_labels = test_features.pop('class')

    return train_features, train_labels, test_features, test_labels


def train_new_model(
        path,
        train_features,
        train_labels,
        ):

    model = create_model(train_features)
    callbacks = compile_model(model, path)
    print(model.summary())
    train_model(model, path, train_features, train_labels, callbacks)

    return model


def test_model(model, test_features, test_labels):
    results = model.evaluate(test_features, test_labels)
    print(f'Test results: {results}')


def create_new_submission(model, path, file='submission.csv'):
    df_submission = open_and_tidy_data(file, submission=True)
    if input('Do you want to analyze the submission dataset? (y/N): ') == 'y':
        analyse_dataset(df_submission, name="submission", submission=True)

    df_submission = parse_data_for_model(df_submission)

    prediction = model.predict(df_submission, batch_size=params.get('batch_size'))

    # model.predict returns a ndarray object, so we must parse it into a 1-dimensional vector
    (pd.Series(prediction[:, 0], name='class')
     .set_axis(df_submission.index)
     .map(lambda x: 'p' if x > 0.7 else 'e')
     .to_csv(f'submissions/{path}.csv'))


def main():
    while True:
        path = input('Load an existing model or leave blank to train a new one: ')

        if path == '':
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = f"{params.get('name')}-{date}"
            print(f'Training a new model with path: {path}')
            train_features, train_labels, test_features, test_labels = load_and_parse_data()
            input('Press enter to start training the model...')
            train_new_model(path, train_features, train_labels)
            break
        elif not os.path.isfile(f'models/{path}.keras'):
            print('Invalid path, please retry')
        else:
            print(f'Using saved model with path: {path}')
            model = tf.keras.models.load_model(f'models/{path}.keras')

            if input('Do you want to test the model? (y/N): ') == 'y':
                train_features, train_labels, test_features, test_labels = load_and_parse_data()
                test_model(model, test_features, test_labels)

            if input('Do you want to generate a submission? (y/N): ') == 'y':
                create_new_submission(model, path)

            break


if __name__ == '__main__':
    main()
