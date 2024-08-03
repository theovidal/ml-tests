import pandas


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

    # We randomly select 80% of the features for the training dataset, so we remove those same features for the testing
    # dataset using the indices (pandas handle this automatically)
    train_dataset = dataset.sample(frac=0.8)
    test_dataset = dataset.drop(train_dataset.index)

    # Analyse the dependencies between the data
    # Here, we mainly want the MPG to be a function of the other parameters
    # sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

    # Statistics shows us that avg and stds are quite high, so we know that we should normalize our dataset
    # so the training would be more stable
    print(train_dataset.describe().transpose())

    # Separate features from label
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    # pop removes a column from the dataset, and returns it
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    return train_features, test_features, train_labels, test_labels

