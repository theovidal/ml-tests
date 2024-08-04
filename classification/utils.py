import datetime
import tensorflow as tf


# Defining global methods to build arbitrary classification models
def get_units_and_activation(num_classes):
    # - One class : useless prediction
    # - Two classes : a binary choice for one of them (in or out)
    # - More classes : we have to give probabilities to belong to each class
    if num_classes > 2:
        return num_classes, 'softmax'
    else:
        return 1, 'sigmoid'


def compile_classification_model(
        model,
        num_classes,
        learning_rate=1e-3,
        early_stopping=2):
    # We are in a classification problem, so we might use other losses (because our probabilities are either solids 1 or
    # 0 in our dataset)
    # If we only have two classes, our probability is straightforward : in or out => binary
    loss = tf.keras.losses.BinaryCrossentropy()
    if num_classes > 2:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Classification problem: we use accuracy as our metric
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    callbacks = []
    if early_stopping is not None:
        # Stop the training early if the validation loss doesn't decrease in 2 consecutive steps
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping))

    return callbacks


def fit_model_tensorboard(model,
                          dataset=None,
                          features=None,
                          labels=None,
                          name='model',
                          verbose=0,
                          epochs=100,
                          validation_split=0.2,
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
        'verbose': verbose,

        # Compute the error using batch_size samples
        'batch_size': batch_size,
        'callbacks': callbacks
    }

    if dataset is None:
        kwargs['validation_split'] = validation_split
        model.fit(
            tf.Variable(features),
            labels,
            **kwargs
        )
    else:
        dataset_size = sum(1 for _ in dataset)
        dataset = dataset.shuffle(buffer_size=1000)
        validation_size = int(validation_split * dataset_size)

        validation_dataset = dataset.take(validation_size)
        train_dataset = dataset.skip(validation_size)
        kwargs['validation_data'] = validation_dataset
        model.fit(train_dataset, **kwargs)

    # Save model
    model.save(f'models/{name}.keras')
    tf.keras.utils.plot_model(model, show_shapes=True)
    print(f'Saved model to models/{name}.keras')
