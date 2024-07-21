import datetime
import pandas
import tensorflow as tf


def get_data():
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    # Column names are not present in this file + separator is different from conventional ","
    raw_dataset = pandas.read_csv('auto-mpg.data.csv', names=column_names,
                                  na_values='?', comment='\t',
                                  sep=' ', skipinitialspace=True)

    # We copy the initial dataset into a new variable in case we want to do destructive operations on it
    dataset = raw_dataset.copy().dropna()

    # Some columns are categories : we choose to one-hot encode them (= associate a vector, where each dimension is a
    # boolean)
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pandas.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')  # Do the conversion

    train_dataset = dataset.sample(frac=0.8)
    test_dataset = dataset.drop(train_dataset.index)  # TODO: explain this instruction

    # Analyse the dependencies between the data
    # Here, we mainly want the MPG to be a function of the other parameters
    # sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

    # Statistics shows us that and stds are quite high, so we know that we should normalize our dataset so the training
    # would be more stable
    print(train_dataset.describe().transpose())

    # Separate features from label
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    # pop removes a column from the dataset, and returns it
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    return train_features, test_features, train_labels, test_labels


def fit_model_tensorboard(model, train_features, train_labels, name='regression', epochs=100, validation_split=0.2):
    # Plug Tensorboard callback so we can better visualize the result
    log_dir = "logs/fit/" + name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        verbose=0,
        # Calculate validation results on 10% of the training data.
        validation_split=validation_split,
        callbacks=[tensorboard_callback]
    )
