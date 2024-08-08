import string

import pandas as pd
from pandas.plotting import scatter_matrix
import dask.dataframe as ddf
import numpy as np
import matplotlib.pyplot as plt

# Assign each column a data type :
# - c for classes
# - b for booleans
# - n for natural numbers
# - f for floating point numbers
# We exclude the "id" key as we don't need it for training
DATASET_COLUMNS = {
    'class': 'c',
    'cap-diameter': 'f',
    'cap-shape': 'c',
    'cap-surface': 'c',  # ID 1675 has a literal value, for no reason...
    'cap-color': 'c',
    'does-bruise-or-bleed': 'b',
    'gill-attachment': 'c',
    'gill-spacing': 'c',
    'gill-color': 'c',
    'stem-height': 'f',
    'stem-width': 'f',
    'stem-root': 'c',
    'stem-surface': 'c',
    'stem-color': 'c',
    'veil-type': 'c',
    'veil-color': 'c',
    'has-ring': 'b',
    'ring-type': 'c',
    'spore-print-color': 'c',
    'habitat': 'c',
    'season': 'c'
}

boolean_mapping = {
    'f': False,
    't': True
}

alphabetic_caracters = list(string.ascii_lowercase)

class_mapping = {
    'e': False,
    'p': True
}


def get_columns(submission=False):
    columns = DATASET_COLUMNS.copy()  # Important to do a copy !
    if submission:
        # The test data doesn't have a class feature (because we must determine it)
        columns.pop('class')
    return columns


# Important topics here:
# - Correctly map data to their type (category, boolean, ...)
# - Verify category names and remove anomalies
# - Correctly handle NaN values depending on the type
def open_and_tidy_data(path, submission=False):
    # TODO: find a way to directly create a sparse matrix
    # data = ddf.read_csv()
    # dtype = pd.SparseDtype(float, fill_value=0)
    # return df.astype(dtype)
    # data = data.map_partitions(lambda part: part.astype(fill_value=0))

    # Creating a sparse matrix to better handle zero values
    df = pd.read_csv(path,
                     na_values=np.nan,
                     skipinitialspace=True)
    df.set_index('id', inplace=True)

    for col, typ in get_columns(submission).items():
        if col == 'class':
            # We want to map the class to a boolean, as we're only applying a binary classification to our dataset
            # (with a threshold)
            df[col] = df[col].map(class_mapping)
        elif typ == 'b':
            df[col] = df[col].map(boolean_mapping)

    # We want our classes to be a unique alphabetic character
    object_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in object_columns:
        df[col] = (
            df[col].apply(lambda x: np.nan if x not in alphabetic_caracters else x)
            .astype('category')
        )

    if not submission:
        # Idiomatic way to shuffle the data :
        # - sample returns a random sample of our dataset, or the fraction is 1, so it returns everything
        # - we reset the index to place the "cursor" to the beginning
        # - we prevent pandas from creating a column indicating the old indexes
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def analyse_dataset(df,
                    name="train",
                    submission=False,
                    hist_nb_bins=30):
    # ------------------
    # STEP 1: get general information about our dataset to try to determine the different values and classes
    # ------------------
    print('--------------------------------------')
    print(f'  DATA ANALYSIS FOR DATASET: {name}  ')
    print('--------------------------------------')
    print(df.info())
    print(df.describe())

    nb_rows = len(df.index)
    nb_columns = len(df.columns)
    nb_plots = nb_columns + 1

    nb_nones = np.array([], dtype=np.int64)

    fig, axes = plt.subplots(nrows=nb_plots, figsize=(8, 6 * nb_plots))
    fig.suptitle(f'Analysis of features from the {name} dataset')

    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        nb_nones = np.append(nb_nones, df[col].isna().sum())

        plt.sca(axes[i])
        # Printing different charts, depending on the type of data
        if dtype == 'category' or dtype == 'bool':
            sizes = df.groupby(by=col).size()
            plt.bar(list(sizes.axes[0]), sizes)
            plt.title(f'Count of categories for column: {col}')
            print(f'Distribution of categories for column {col}')
            print(df[col].value_counts() / nb_rows)
        else:
            plt.hist(df[col], bins=hist_nb_bins)
            plt.grid(True)
            plt.title(f'Distribution of values for column: {col}')

            # Useless: already done by df.describe()
            # print(f'------- STATISTICS FOR FEATURE: {col}')
            # print(f'- Minimum: {df_col.min()}')
            # print(f'- Maximum: {df_col.max()}')
            # print(f'- Mean: {df_col.mean()}')
            # print(f'- Median: {df_col.median()}')
            # print(f'- Standard deviation: {df_col.std()}')

    # Printing the proportion of unset values for each column
    plt.sca(axes[nb_columns])
    plt.bar(list(df.columns), nb_nones / nb_rows)
    plt.xlabel('Column names')
    plt.ylabel('Proportion of nan values')
    plt.xticks(rotation='vertical')
    plt.grid(True)

    fig.tight_layout()

    # ------------------
    # STEP 2: more detailed statistics to detect correlations
    # ------------------

    print('---------- Correlation matrix ----------')
    # We want to compute correlation between categorical features, so we must first factorize them
    # "factorize" = transform object to enum or categorical features
    df_fact = df.apply(lambda x: pd.factorize(x)[0])
    corr_matrix = df_fact.corr(method='pearson', min_periods=1)
    print(corr_matrix)
    # Consumes too much memory
    # scatter_matrix(df_fact, figsize=(8 * nb_columns, 6 * nb_columns))

    if not submission:
        print('---------- Edible correlations ----------')
        # For each feature, we want to know :
        # - the proportion of each category being poisonous,
        # - or the mean value of each class (p/e)

        cor_fig, cor_axes = plt.subplots(nrows=nb_columns, figsize=(8, 6 * nb_columns))

        for i, col in enumerate(df.columns):
            dtype = df[col].dtype

            plt.sca(cor_axes[i])
            # TODO: see for color
            # cor_axes[i].set_prop_cycle(color=list(matplotlib.colors.TABLEAU_COLORS.keys()))
            group = None
            if dtype == 'category' or dtype == 'bool':
                group = df.groupby(col)['class'].mean()
                plt.title(f'Proportion of poisonous mushrooms for column {col}')
            else:
                group = df.groupby('class')[col].mean()
                plt.title(f'Average values of column {col} by consumption type')
            plt.bar(group.axes[0], group)

        cor_fig.tight_layout()

        # This is complicated for nothing...
        # for i, col in enumerate(df.columns):
        #     if col == 'class':
        #         continue
        #
        #     dtype = df[col].dtype
        #
        #     if dtype == 'category' or dtype == 'boolean':
        #         count_class_from_column(df, col).plot(kind='bar',
        #                                               title=f'Edible correlation of categorical feature {col}')
        #     else:
        #         df.groupby('class')[col].plot(kind='hist', title=f'Edible correlation of categorical feature {col}')

    plt.show()


def parse_data_for_model(df):
    # Fill NaN values with either a special category, indicating the absence of value, or the median of all values
    # In order to handle different categories in train and submission dataset, we fill all the categories by default
    for col in df.columns:
        typ = df[col].dtype
        if typ == 'category':
            df[col] = df[col].cat.set_categories(alphabetic_caracters)
            df[col] = df[col].cat.add_categories(['n/a'])
            df[col] = df[col].fillna('n/a')
        elif typ == 'bool':
            df[col] = df[col].map(lambda x: 1 if x else 0)
        else:
            df[col] = df[col].fillna(df[col].median())

    # Normalization of numerical values is already included in the model as the first layer

    # One-hot encoding all the variables with type "category"
    # Except for the "class" which will be extracted (it's not a feature but the label)
    return pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'category'], dtype=float)

    # dtype = pd.SparseDtype(float, fill_value=0)
    # return df.astype(dtype)
