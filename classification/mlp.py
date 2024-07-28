import tensorflow as tf

from utils import fit_model_tensorboard

LEARNING_RATE = 1e-3
HIDDEN_LAYERS = 2
EPOCHS = 1000


# Defining global methods to build arbitrary classification models
def get_units_and_activation(num_classes):
    # - One class : useless prediction
    # - Two classes : a binary choice for one of them (in or out)
    # - More classes : we have to give probabilities to belong to each class
    if num_classes > 2:
        return num_classes, 'softmax'
    else:
        return 1, 'sigmoid'


def create_sequential(num_layers, units, activation, num_classes, input_shape, dropout_rate, normalization=None):
    op_units, op_activation = get_units_and_activation(num_classes)

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


def mlp_model(num_classes, x_train, y_train, x_validation, y_validation):
    # Input shape is a (n, m) matrix where :
    # - n is the number of samples
    # - m is the vocabulary size (so the number of probabilities)
    # Hence the input shape of our model = the vocabulary size
    model = create_sequential(HIDDEN_LAYERS, 64, 'relu', num_classes, x_train.shape[1:], 0.2)

    # We are in a classification problem, so we might use other losses (because our probabilities are either solids 1 or
    # 0 in our dataset)
    # If we only have two classes, our probability is straightforward : in or out => binary
    loss = 'binary_crossentropy'
    if num_classes > 2:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Classification problem: we use accuracy as our metric
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    print(model.summary())

    # Stop the training early if the validation loss doesn't decrease in 2 consecutive steps
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    return fit_model_tensorboard(
        model,
        x_train,
        y_train,
        'imdb_classification_mlp',
        batch_size=512,
        epochs=EPOCHS,
        validation_data=(x_validation, y_validation),
        callbacks=[early_stop]
    )
