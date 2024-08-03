import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

from data import get_data

np.set_printoptions(precision=3, suppress=True)


# Compile a model with any number of inputs (that are included in the normalization layer whatsoever)
def build_and_compile_model(nb_layers, units, norm=None, activation='relu', learning_rate=0.01, dropout_rate=0.0):
    model = tf.keras.Sequential()

    if norm is not None:
        model.add(norm)
    model.add(tf.keras.layers.Dropout(dropout_rate))

    for _ in range(nb_layers):
        model.add(tf.keras.layers.Dense(units, activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model


def fit_model_tensorboard(model,
                          train_features,
                          train_labels,
                          name='model',
                          epochs=100,
                          validation_split=0.2,
                          validation_data=None,
                          batch_size=None,
                          callbacks=None):
    # Plug Tensorboard callback so we can better visualize the result

    # Trick given by PEP to have immutable function arguments, and then fill with what we want
    if callbacks is None:
        callbacks = []

    log_dir = "logs/fit/" + name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    kwargs = {
        'epochs': epochs,
        'verbose': 0,

        # Compute the error using batch_size samples
        'batch_size': batch_size,
        'callbacks': callbacks
    }

    if validation_data is not None:
        kwargs['validation_data'] = validation_data
    else:
        kwargs['validation_split'] = validation_split

    return model.fit(
        train_features,
        train_labels,
        **kwargs
    )


def main():
    train_features, test_features, train_labels, test_labels = get_data()

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    model = build_and_compile_model(2, 64, normalizer, learning_rate=0.001, dropout_rate=0.0)
    print(model.summary())

    history = fit_model_tensorboard(model, train_features, train_labels, name='nn', epochs=300)

    print("Test loss = ", model.evaluate(test_features, test_labels, verbose=0))


if __name__ == '__main__':
    main()
    plt.show()
