import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import data
from .. import utils

np.set_printoptions(precision=3, suppress=True)


# Manually plot the loss from the history
# Use of tensorboard to visualize data is recommended
def plot_loss(history):
    plt.figure()
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


def main():
    train_features, test_features, train_labels, test_labels = data.get_data()

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units=1)  # We apply a simple transformation : y = mx + b
    ])

    # Weights are generated when the first prediction is asked ; the shape of the weights will exactly match the shape
    # of the input
    # (Outputs are of course wrong here, as the model isn't trained at all)
    model.predict(train_features[:10])

    print(model.summary())

    # We can verify that our weights have the correct shape (here, we have 9 features and 1 label so (9, 1)
    print('Shape of the model: ', model.layers[1].kernel)

    # Compile our model to then train it on our training dataset
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    history = utils.fit_model_tensorboard(model, features=train_features, labels=train_labels)

    print("Test loss = ", model.evaluate(test_features, test_labels, verbose=0))

    # If we have exactly one input and one output : we can plot the result
    # x = tf.linspace(0.0, 250, 251)
    # y = model.predict(x)

    # plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    # plt.plot(x, y, color='k', label='Predictions')
    # plt.xlabel('Horsepower')
    # plt.ylabel('MPG')
    # plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
