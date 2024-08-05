import tensorflow as tf
import utils


def create_sequential(num_layers, units, activation, num_classes, input_shape, dropout_rate, normalization=None):
    op_units, op_activation = utils.get_units_and_activation(num_classes)

    # Simpler to use: add layers as they are created
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape, sparse=True))
    # model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    if normalization is not None:
        model.add(normalization)

    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
        # We should add a Dropout layer here to catch examples before they get to the output
        # model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    model.add(tf.keras.layers.Dense(units=op_units, activation=op_activation))
    return model


def mlp_model(num_classes,
              train_features,
              train_labels,
              layers=2,
              epochs=1000,
              learning_rate=1e-3):
    # Input shape is a (n, m) matrix where :
    # - n is the number of samples
    # - m is the vocabulary size (so the number of probabilities)
    # Hence the input shape of our model = the vocabulary size
    model = create_sequential(layers, 64, 'relu', num_classes, train_features.shape[1:], 0.2)

    callbacks = utils.compile_classification_model(model, num_classes)

    utils.fit_model_tensorboard(model, features=train_features, labels=train_labels, callbacks=callbacks)
