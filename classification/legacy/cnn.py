import tensorflow as tf

import utils
from params import model_params


def get_text_vectorizer(vocabulary):
    return tf.keras.layers.TextVectorization(
        vocabulary=vocabulary,
        output_mode='int',
        output_sequence_length=model_params["max_sequence_length"]
    )


def init_cnn_model(
        vocabulary,
        num_classes):

    # ------------------
    # STEP 1: tokenize our dataset to transform it into word sequences (those words are mapped to integers)
    # ------------------
    # This is done using the vectorization layer, which can't be implemented here (only available to Functional API)
    # Instead, we do the vectorization outside

    # The input shape will be deduced when the model will be built, i.e. when we will compile and evaluate the model
    # for the first time (because it depends on the chosen sequence length)
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    # ------------------
    # STEP 2: create the word embedding so the model can semantically interpret our sentences
    # ------------------
    num_features = len(vocabulary)
    x = tf.keras.layers.Embedding(
        num_features,
        model_params["embedding_dim"],
        mask_zero=True,  # This option serves to ignore zeros in the padding (after having extended sequences to match
                         # length) when passing to convolution blocks, or more efficiently to RNN blocks
    )(inputs)

    for i in range(model_params["num_blocks"]):
        x = tf.keras.layers.Dropout(rate=model_params["dropout_rate"])(x)
        x = tf.keras.layers.SeparableConv1D(
            kernel_size=model_params["kernel_size"],
            filters=model_params["num_filters"],
            padding="same",
            activation=model_params["activation"],
            depthwise_initializer=tf.keras.initializers.RandomUniform,
            bias_initializer=tf.keras.initializers.RandomUniform
        )(x)
        if i < model_params["num_blocks"] - 1:
            x = tf.keras.layers.MaxPool1D(
                pool_size=model_params["pool_size"]
            )(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(rate=model_params["dropout_rate"])(x)

    # Every model should have a dense layer at the end, in order to return a coherent result
    op_units, op_activation = utils.get_units_and_activation(num_classes)
    outputs = tf.keras.layers.Dense(units=op_units, activation=op_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def train_new_cnn_model(vocabulary,
                        num_classes,
                        train_texts,
                        train_labels):

    model = init_cnn_model(vocabulary, num_classes)
    text_vectorizer = get_text_vectorizer(vocabulary)

    # We create a unique dataset for our two components : features and labels
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    # We then use a pre-processing pipeline which treatment will be optimized by TensorFlow
    train_dataset = train_dataset.batch(model_params["batch_size"]).map(lambda x, y: (text_vectorizer(x), y))

    # If the model doesn't contain a preprocessing layer (so, using the Functional API) we must
    # transform our dataset outside, and then pass it
    callbacks = utils.compile_classification_model(model, num_classes)

    utils.fit_model_tensorboard(model, dataset=train_dataset, callbacks=callbacks)
    print("Model trained. Check statistics on TensorBoard using the logs/fit directory.")
    return model
