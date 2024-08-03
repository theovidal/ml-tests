import tensorflow as tf
import numpy as np

import utils


def get_vectorize_layer(vocabulary, max_sequence_length=500):
    return tf.keras.layers.TextVectorization(
        vocabulary=vocabulary,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )


def create_cnn(
        vocabulary,
        num_classes,
        embedding_dim=200,
        num_filters=64,
        num_blocks=2,
        dropout_rate=0.2,
        kernel_size=3,
        pool_size=3,
        max_sequence_length=500,
        activation=tf.keras.activations.relu):

    model = tf.keras.Sequential()

    # ------------------
    # STEP 1: tokenize our dataset to transform it into word sequences (those words are mapped to integers)
    # ------------------
    # This is done using the vectorization layer, which can't be implemented here (only available to Functional API)
    # Instead, we do the vectorization outside

    # ------------------
    # STEP 2: create the word embedding so the model can semantically interpret our sentences
    # ------------------
    num_features = len(vocabulary)
    model.add(tf.keras.layers.Embedding(
        num_features,
        embedding_dim,
        input_length=max_sequence_length,
    ))

    for i in range(num_blocks):
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        # Separable convolution combines a depth-wise convolution, followed by a point-wise convolution
        model.add(tf.keras.layers.SeparableConv1D(
            # Size of the convolution windows that runs independently on every channel (depth-wise part)
            kernel_size=kernel_size,
            # Total number of filters extracted, i.e. the output size (point-wise part)
            filters=num_filters,
            padding="same",
            activation=activation,
            depthwise_initializer=tf.keras.initializers.RandomUniform,
            bias_initializer=tf.keras.initializers.RandomUniform
        ))
        if i < num_blocks - 1:
            model.add(tf.keras.layers.MaxPool1D(
                pool_size=pool_size
            ))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Every model should have a dense layer at the end, in order to return a coherent result
    op_units, op_activation = utils.get_units_and_activation(num_classes)
    model.add(tf.keras.layers.Dense(units=op_units, activation=op_activation))

    return model


def cnn_model(vocabulary,
              num_classes,
              train_texts,
              train_labels,
              learning_rate=1e-3,
              epochs=1000):

    model = create_cnn(vocabulary, num_classes)
    vectorize_layer = get_vectorize_layer(vocabulary)

    # If the model doesn't contain a preprocessing layer (so, using the Functional API) we must
    # transform our dataset outside, and then pass it
    x_train = vectorize_layer(train_texts)
    callbacks = utils.compile_classification_model(model, num_classes, learning_rate)

    # We choose to build the model before fitting, so we can preview all the parameters
    model.evaluate(x_train[:1], train_labels[:1])
    model.summary()

    accuracy, loss = utils.fit_model_tensorboard(
        model,
        x_train,
        train_labels,
        name='imdb_classification_cnn',
        batch_size=128,
        epochs=epochs,
        callbacks=callbacks
    )
    print(f"Accuracy during training: {accuracy}\nLoss: {loss}")

    return model
