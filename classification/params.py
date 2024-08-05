model_params = {
    'name': 'imdb_classification_cnn',

    # ---------- Model shape ----------
    'embedding_dim': 200,
    'num_blocks': 3,
    'max_sequence_length': 500,

    # ---------- Model parameters ----------
    'activation': 'relu',
    'dropout_rate': 0.35,

    # ---------- Convolution parameters ----------
    # Separable convolution combines a depth-wise convolution, followed by a point-wise convolution
    'kernel_size': 3,  # Size of the convolution windows that runs independently on every channel (depth-wise part)
    'num_filters': 64,  # Total number of filters extracted, i.e. the output size (point-wise part)
    'pool_size': 3,  # For the Max pooling

    # Training parameters
    'learning_rate': 1e-3,
    'epochs': 1000,
    'batch_size': 128,  # Number of sample blocks used (each block is passed at once)
    'validation_split': 0.2,  # Proportion of the dataset used for validation
    'early_stopping': 2,  # Stop when the loss is constant for _ consecutive steps ; set to None to disable

    # Debugging and analysing
    'verbose': 0,
    'log_dir': 'logs/fit'

}
