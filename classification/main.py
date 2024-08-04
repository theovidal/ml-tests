# Based on https://developers.google.com/machine-learning/guides/text-classification
import os.path
import datetime
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from open_data import load_imdb_sentiment_analysis_dataset
from analyse_data import analyse_dataset
from mlp import mlp_model
from cnn import cnn_model, get_vectorize_layer
from vectorize_sequences import get_vocabulary
from vectorize_ngram import vectorize_ngrams
from utils import compile_classification_model


def execute_mlp(train_texts, train_labels, test_texts, test_labels, classes_list):
    x_train, x_text = vectorize_ngrams(train_texts, train_labels, test_texts)

    accuracy, loss = mlp_model(len(classes_list), x_train, train_labels, x_text, test_labels)
    print(f"Accuracy: {accuracy}\nLoss: {loss}")


def execute_cnn(train_texts, train_labels, test_texts, test_labels, classes_list):
    model = None
    vocabulary = get_vocabulary(train_texts)
    vectorize_layer = get_vectorize_layer(vocabulary)

    if os.path.isfile("models/imdb_classification_cnn.keras"):
        print("Using saved model")
        model = tf.keras.models.load_model("models/imdb_classification_cnn.keras")
    else:
        print("No saved model: training a new one")
        model = cnn_model(vocabulary, len(classes_list), train_texts, train_labels)

    # Making an end-to-end model that includes the preprocessing layer, which is more convenient for
    # inference.
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    x = vectorize_layer(inputs)
    end_to_end_model = tf.keras.Model(inputs=inputs, outputs=model(x))

    log_dir = "logs/evaluate/cnn-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    end_to_end_model.predict(tf.constant(test_texts[:1]))

    # To evaluate our model, we must still use the other version, that was saved with the compilation information
    # and so the metrics are built
    x_train = vectorize_layer(test_texts)
    loss = model.evaluate(tf.constant(x_train),
                          test_labels,
                          callbacks=[tensorboard_callback])
    print(f"Test loss: {loss}")


def main():
    train_texts, train_labels, train_classes, test_texts, test_labels, test_classes = (
        load_imdb_sentiment_analysis_dataset('aclImdb', int(time.time())))
    nb_samples, classes_list, word_counts, avg_nb_words = analyse_dataset(train_texts, train_labels, train_classes)

    while True:
        choice = input("Which model to use: MLP (choice 1) or CNN (choice 2)? ")
        if choice == '1':
            execute_mlp(train_texts, train_labels, test_texts, test_labels, classes_list)
            break
        elif choice == '2':
            execute_cnn(train_texts, train_labels, test_texts, test_labels, classes_list)
            break
        else:
            print("Invalid input. Please enter 1 or 2.")


if __name__ == '__main__':
    # ---------- Setting options before executing our program ----------
    np.set_printoptions(precision=3, suppress=True)
    # TF_GPU_ALLOCATOR=cuda_malloc_async
    # tf.debugging.set_log_device_placement(True)

    # TODO: correct the GPU bug
    # tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))

    # ----------
    main()
    # ----------

    # ---------- Plotting and cleanup ----------
    plt.show()
