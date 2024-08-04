import tensorflow as tf

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
        activation=tf.keras.activations.relu):

    # ------------------
    # STEP 1: tokenize our dataset to transform it into word sequences (those words are mapped to integers)
    # ------------------
    # This is done using the vectorization layer, which can't be implemented here (only available to Functional API)
    # Instead, we do the vectorization outside

    # The input shape will be deduced when the model will be built, i.e when we will compile and evaluate the model
    # for the first time (because it depends on the chosen sequence length)
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # ------------------
    # STEP 2: create the word embedding so the model can semantically interpret our sentences
    # ------------------
    num_features = len(vocabulary)
    x = tf.keras.layers.Embedding(
        num_features,
        embedding_dim,
    )(inputs)

    for i in range(num_blocks):
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        # Separable convolution combines a depth-wise convolution, followed by a point-wise convolution
        x = tf.keras.layers.SeparableConv1D(
            # Size of the convolution windows that runs independently on every channel (depth-wise part)
            kernel_size=kernel_size,
            # Total number of filters extracted, i.e. the output size (point-wise part)
            filters=num_filters,
            padding="same",
            activation=activation,
            depthwise_initializer=tf.keras.initializers.RandomUniform,
            bias_initializer=tf.keras.initializers.RandomUniform
        )(x)
        if i < num_blocks - 1:
            x = tf.keras.layers.MaxPool1D(
                pool_size=pool_size
            )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    # Every model should have a dense layer at the end, in order to return a coherent result
    op_units, op_activation = utils.get_units_and_activation(num_classes)
    outputs = tf.keras.layers.Dense(units=op_units, activation=op_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def cnn_model(vocabulary,
              num_classes,
              train_texts,
              train_labels,
              learning_rate=1e-3,
              epochs=1000):

    model = create_cnn(vocabulary, num_classes)
    vectorize_layer = get_vectorize_layer(vocabulary)

    # We create a unique dataset for our two components : features and labels
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    # We then use a pre-processing pipeline which treatment will be optimized by TensorFlow
    train_dataset = train_dataset.batch(128).map(lambda x, y: (vectorize_layer(x), y))

    # If the model doesn't contain a preprocessing layer (so, using the Functional API) we must
    # transform our dataset outside, and then pass it
    callbacks = utils.compile_classification_model(model, num_classes, learning_rate)

    utils.fit_model_tensorboard(model,
                                dataset=train_dataset,
                                name='imdb_classification_cnn',
                                epochs=epochs,
                                batch_size=128,
                                callbacks=callbacks)
    print("Model trained. Check statistics on TensorBoard using the logs/fit directory.")
    return model
