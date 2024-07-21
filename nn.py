import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils

np.set_printoptions(precision=3, suppress=True)


# Compile a model with any number of inputs (that are included in the normalization layer whatsoever)
def build_and_compile_model(norm, learning_rate=0.01):
    model = tf.keras.Sequential([
        norm,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model


def main():
    train_features, test_features, train_labels, test_labels = utils.get_data()

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    model = build_and_compile_model(normalizer, learning_rate=0.001)
    print(model.summary())

    history = utils.fit_model_tensorboard(model, train_features, train_labels, name='nn', epochs=300)

    print("Test loss = ", model.evaluate(test_features, test_labels, verbose=0))


if __name__ == '__main__':
    main()
    plt.show()
