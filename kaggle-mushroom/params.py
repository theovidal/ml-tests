import os

default_params = {
    # Model shape
    'name': 'model',
    'nb_layers': 2,
    'nb_units': 1024,
    'dropout_rate': 0.2,

    # Model parameters
    'activation': 'relu',
    'threshold': 0.4,

    # Training step
    'epochs': 300,
    'learning_rate': 0.001,
    'early_stopping': 2,
    'validation_split': 0.2,
    'batch_size': 128,

    # Debugging
    'verbose': 0,
    'log_dir': 'logs/fit'
}


def get(name):
    if name not in default_params.keys():
        raise KeyError(f'{name} is not a model parameter')

    return os.environ.get(name, default_params[name])
