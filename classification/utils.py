import datetime
import json

import tensorflow as tf

from params import model_params


def get_units_and_activation(num_classes):
    """
        - One class : useless prediction
        - Two classes : a binary choice for one of them (in or out)
        - More classes : we have to give probabilities to belong to each class
    :param num_classes:
    :return: a tuple of (number of outputs, last activation function)
    """

    if num_classes > 2:
        return num_classes, 'softmax'
    else:
        return 1, 'sigmoid'


def compile_classification_model(
        model,
        num_classes):
    """
    Compile a model specifically for classification, using a loss fitted for the problem, and an Adam optimizer

    :param model:
    :param num_classes:
    :return: an array of callbacks to plug when fitting, i.e. early stopping
    """

    # We are in a classification problem, so we might use other losses (because our probabilities are either solids 1 or
    # 0 in our dataset)
    # If we only have two classes, our probability is straightforward : in or out => binary
    loss = tf.keras.losses.BinaryCrossentropy()
    if num_classes > 2:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(model_params["learning_rate"])

    # Classification problem: we use accuracy as our metric
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    callbacks = []
    if model_params["early_stopping"] is not None:
        # Stop the training early if the validation loss doesn't decrease in 2 consecutive steps
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=model_params["early_stopping"]))

    return callbacks


def fit_model_tensorboard(model,
                          dataset=None,
                          features=None,
                          labels=None,
                          callbacks=None):
    """
    Fit the model using either:
        - a given dataset
        - separate features and labels
    Each one of them will be split into training and validation sets
    All the data will then be exported for TensorBoard analysis

    :param callbacks: additional callbacks in addition to TensorBoard
    :param labels:
    :param features:
    :param dataset: tf.data.Dataset object
    :param model The Keras model to fit
    """

    # Trick given by PEP to have immutable function arguments, and then fill with what we want
    if callbacks is None:
        callbacks = []

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = f'{model_params["log_dir"]}/{model_params["name"]}-{date}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    kwargs = {
        'epochs': model_params["epochs"],
        'verbose': model_params["verbose"],

        # 'batch_size': model_params["batch_size"],
        'callbacks': callbacks
    }

    if dataset is None:
        kwargs['validation_split'] = model_params["validation_split"]
        model.fit(
            tf.Variable(features),
            labels,
            **kwargs
        )
    else:
        dataset_size = sum(1 for _ in dataset)
        dataset = dataset.shuffle(buffer_size=1000)
        validation_size = int(model_params["validation_split"] * dataset_size)

        validation_dataset = dataset.take(validation_size)
        train_dataset = dataset.skip(validation_size)
        kwargs['validation_data'] = validation_dataset
        model.fit(train_dataset, **kwargs)

    # Save model and parameters
    model.save(f'models/{model_params["name"]}-{date}.keras')
    with open(f'models/{model_params["name"]}-{date}.json', 'w') as file:
        json.dump(model_params, file)
    tf.keras.utils.plot_model(model, show_shapes=True)
    print(f'Saved model to models/{model_params["name"]}.keras')
