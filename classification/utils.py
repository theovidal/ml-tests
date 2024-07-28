import datetime

import tensorflow as tf


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
        train_features.toarray(),
        train_labels,
        **kwargs
    )
