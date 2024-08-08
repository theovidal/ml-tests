import tensorflow as tf
import numpy as np

import params


def create_model(train_features):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    nb_inputs = len(train_features.columns)

    inputs = tf.keras.Input(shape=(nb_inputs,))
    x = normalizer(inputs)
    # For each layer, don't forget to pass the preceding one!
    x = tf.keras.layers.Dropout(rate=params.get('dropout_rate'))(x)

    for i in range(params.get('nb_layers')):
        x = tf.keras.layers.Dense(units=params.get('nb_units'), activation=params.get('activation'))(x)
        x = tf.keras.layers.Dropout(rate=params.get('dropout_rate'))(x)

    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def compile_model(
        model,
        path,
        callbacks=None):

    if callbacks is None:
        callbacks = []

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate'))
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=params.get('threshold')),
        tf.keras.metrics.Precision(name='precision', thresholds=params.get('threshold')),
        tf.keras.metrics.Recall(name='recall', thresholds=params.get('threshold'))
    ]

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=metrics)

    if params.get('early_stopping') is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=params.get('early_stopping')))

    log_dir = f"{params.get('log_dir')}/{path}"
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    return callbacks


def train_model(model,
                path,
                train_features,
                train_labels,
                callbacks=None,
                ):

    if callbacks is None:
        callbacks = []

    model.fit(
        train_features,
        train_labels,
        epochs=params.get('epochs'),
        validation_split=params.get('validation_split'),
        batch_size=params.get('batch_size'),
        callbacks=callbacks
    )
    model.save(f"models/{path}.keras")

    tf.keras.utils.plot_model(model, show_shapes=True)
    print(f"Saved model to models/{path}.keras")

