import pandas as pd
import numpy as np

"""
Objects and Methods for Iterative ML Development Series with Titanic Data Set
"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

# enable and import new iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import BayesianRidge


def print_scores(scores):
    """"Print CV Scores in a standard format"""

    print(f'{len(scores)} Scores  min:{scores.min():.3f} max:{scores.max():.3f}')
    print(f'CV Mean Score: {scores.mean():.3f} +/- {scores.std():.3f}')


def print_grid(grid, pandas=False):
    """Print Best and Return Results in a DataFrame"""

    sd = grid.cv_results_['std_test_score'][grid.best_index_]
    print(f'Best: {grid.best_score_:0.3f} +/- {sd:0.3f}')
    for key, value in grid.best_params_.items():
        print(f'{key}: {value}')

    if pandas:
        results = []
        for i in range(len(grid.cv_results_['mean_test_score'])):
            score = grid.cv_results_['mean_test_score'][i]
            std = grid.cv_results_['std_test_score'][i]
            params = grid.cv_results_['params'][i]
            params['score'] = score
            params['std'] = std
            results.append(params)

        return pd.DataFrame(results)


def get_ct_v1():
    """Column Transform for Features

    Version 1
    * without Categorical Variable Encoding
    * uses SimpleImputer for Age

    Returns column names and ColumnTransform instance.
    """
    
    # For numeric columns
    ss = StandardScaler()

    # To impute age
    si = SimpleImputer()

    # quantize fare to below/above median
    kbin = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')

    # Pipelines
    ss_pipe = Pipeline([('ss', ss)])
    ss_si_pipe = Pipeline([('ss', ss), ('si', si)])
    kbin_pipe = Pipeline([('kbin', kbin)])

    # Columns to act on
    ss_cols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'family_size']
    ss_si_cols = ['Age']
    kbin_cols = ['Fare']
    bool_cols = ['Sex', 'is_cabin_notnull', 'is_large_family', 'is_child', 
                 'is_sibsp_zero', 'is_parch_zero', 'is_boy']

    transformers = [('ss_tr', ss_pipe, ss_cols),
                    ('ss_si_tr', ss_si_pipe, ss_si_cols),
                    ('kbin_tr', kbin_pipe, kbin_cols),
                    ('as_is', 'passthrough', bool_cols)]

    ct = ColumnTransformer(transformers=transformers)

    # there is no way to access the columns by name from a pipe
    # create a list of columns to keep track
    cols = ss_cols + ss_si_cols + ['is_fare_high'] + bool_cols

    return cols, ct


class WrappedIterativeImputer(BaseEstimator, TransformerMixin):
    """Wrap IterativeImputer to return One Column Only

    The name of the column to be kept is passed to the constructor.

    This must be the first step in a pipe, as it relies on X being a Pandas DataFrame"""

    def __init__(self, return_col):
        self.ii = IterativeImputer()
        self.return_col = return_col
        self.return_col_idx = None

    def fit(self, X, y):
        self.ii.fit(X, y)
        self.return_col_idx = X.columns.get_loc(self.return_col)

        return self

    def transform(self, X):
        return_col = self.ii.transform(X)[:, self.return_col_idx]

        # must be 2D
        return return_col.reshape(-1, 1)


def get_ct_v2():
    """Column Transform for Features

    Version 2
    * without Categorical Variable Encoding
    * uses Wrapped IterativeImputer for Age

    The IterativeImputer needs many columns in order to impute well.

    Returns column names and ColumnTransform instance.
    """

    # For numeric columns
    ss = StandardScaler()

    # Only used to impute Age
    ii = WrappedIterativeImputer('Age')

    # quantize fare to below/above median
    kbin = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')

    # Pipelines
    ss_pipe = Pipeline([('ss', ss)])

    # wrapped IterativeImputer uses many columns, but only outputs Age
    # which is then standardized
    ii_ss_pipe = Pipeline([('ii', ii), ('ss', ss)])

    # quantize Fare to below/above median
    kbin_pipe = Pipeline([('kbin', kbin)])

    # Columns to act on
    ss_cols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'family_size']
    ii_ss_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                  'family_size', 'is_cabin_notnull', 'is_large_family',
                  'is_child', 'is_sibsp_zero', 'is_parch_zero', 'is_boy']
    kbin_cols = ['Fare']
    bool_cols = ['Sex', 'is_cabin_notnull', 'is_large_family', 'is_child', 
                 'is_sibsp_zero', 'is_parch_zero', 'is_boy']

    transformers = [('ss_tr', ss_pipe, ss_cols),
                    ('ii_SS_tr', ii_ss_pipe, ii_ss_cols),
                    ('kbin_tr', kbin_pipe, kbin_cols),
                    ('as_is', 'passthrough', bool_cols)]

    # instantiate ColumnTransformer
    ct = ColumnTransformer(transformers=transformers)

    # there is no way to access the columns by name from a pipe
    # create a list of columns to keep track
    cols = ss_cols + ['Age'] + ['is_fare_high'] + bool_cols

    return cols, ct


def get_ct_v3():
    """Column Transform for Features

    Version 3
    * with Categorical Variable Encoding
    * uses all columns for Wrapped IterativeImputer

    Returns column names and ColumnTransform instance.
    """

    ss = StandardScaler()
    ii = WrappedIterativeImputer('Age')
    kbin = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')

    # Pipelines
    ss_pipe = Pipeline([('ss', ss)])
    ii_ss_pipe = Pipeline([('ii', ii), ('ss', ss)])
    kbin_pipe = Pipeline([('kbin', kbin)])

    # Columns to act on
    ss_cols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'family_size']
    ii_ss_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'family_size',
                  'is_cabin_notnull', 'is_large_family', 'is_child', 'is_sibsp_zero',
                  'is_parch_zero', 'is_boy', 'Port_C', 'Port_Q', 'Port_S', 
                  'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']
    kbin_cols = ['Fare']
    bool_cols = ['Sex', 'is_cabin_notnull', 'is_large_family', 'is_child', 
                 'is_sibsp_zero', 'is_parch_zero', 'is_boy',
                 'Port_C', 'Port_Q', 'Port_S', 'Title_Master',
                 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

    transformers = [('ss_tr', ss_pipe, ss_cols),
                    ('ii_ss_tr', ii_ss_pipe, ii_ss_cols),
                    ('kbin_tr', kbin_pipe, kbin_cols),
                    ('as_is', 'passthrough', bool_cols)]

    ct = ColumnTransformer(transformers=transformers)

    # there is no way to access the columns by name from a pipe
    # create a list of columns to keep track
    cols = ss_cols + ['Age'] + ['is_fare_high'] + bool_cols

    return cols, ct


def get_ct_v4():
    """Column Transform for Features

    Version 4
    * with Categorical Variable Encoding
    * use subset of variables for Wrapped IterativeImputer

    Returns column names and ColumnTransform instance.
    """

    ss = StandardScaler()
    ii = WrappedIterativeImputer('Age')
    kbin = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')

    # Pipelines
    ss_pipe = Pipeline([('ss', ss)])
    ii_ss_pipe = Pipeline([('ii', ii), ('ss', ss)])
    kbin_pipe = Pipeline([('kbin', kbin)])

    # Columns to act on
    ss_cols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'family_size']
    ii_ss_cols = ['Pclass', 'Sex', 'Age', 'Title_Master',
                  'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']
    kbin_cols = ['Fare']
    bool_cols = ['Sex', 'is_cabin_notnull', 'is_large_family', 'is_child',
                 'is_sibsp_zero', 'is_parch_zero', 'is_boy',
                 'Port_C', 'Port_Q', 'Port_S', 'Title_Master',
                 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

    transformers = [('ss_tr', ss_pipe, ss_cols),
                    ('ii_ss_tr', ii_ss_pipe, ii_ss_cols),
                    ('kbin_tr', kbin_pipe, kbin_cols),
                    ('as_is', 'passthrough', bool_cols)]

    ct = ColumnTransformer(transformers=transformers)

    # there is no way to access the columns by name from a pipe
    # create a list of columns to keep track
    cols = ss_cols + ['Age'] + ['is_fare_high'] + bool_cols

    return cols, ct


def get_ct_v5():
    """Column Transform for Features

    Version 5
    * with Categorical Variable Encoding
    * use subset of variables for Wrapped IterativeImputer
    * use subset of variables for prediction

    Returns column names and ColumnTransform instance.
    """

    ss = StandardScaler()
    ii = WrappedIterativeImputer('Age')

    # Pipelines
    ss_pipe = Pipeline([('ss', ss)])
    ii_ss_pipe = Pipeline([('ii', ii), ('ss', ss)])

    # Columns to act on
    ss_cols = ['Pclass', 'Fare', 'family_size']
    ii_ss_cols = ['Pclass', 'Sex', 'Age', 'Title_Master',
                  'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']
    bool_cols = ['Sex', 'is_cabin_notnull',
                 'Port_C', 'Port_Q', 'Port_S', 'Title_Master',
                 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

    transformers = [('ss_tr', ss_pipe, ss_cols),
                    ('ii_ss_tr', ii_ss_pipe, ii_ss_cols),
                    ('as_is', 'passthrough', bool_cols)]

    ct = ColumnTransformer(transformers=transformers)

    # there is no way to access the columns by name from a pipe
    # create a list of columns to keep track
    cols = ss_cols + ['Age'] + bool_cols

    return cols, ct


def get_Xy_v1(filename='./data/train.csv'):
    """Data Encoding

    Version 1
    * Pclass and Sex encoded as 1/0
    """

    # read data
    all_data = pd.read_csv(filename)
    X = all_data.drop('Survived', axis=1)
    y = all_data['Survived']
    
    # encode data
    X['Sex'] = X['Sex'].replace({'female':1, 'male':0})
    
    # drop unused columns
    drop_columns = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 
                    'Fare', 'Ticket', 'Cabin', 'Embarked']
    X = X.drop(drop_columns, axis=1)
    
    return X, y


def get_Xy_v2(filename='./data/train.csv'):
    """Data Encoding

    Version 2
    * Pclass and Sex encoded as 1/0
    * Age, Fare, SibSp, Parch
    """

    # read data
    all_data = pd.read_csv(filename)
    X = all_data.drop('Survived', axis=1)
    y = all_data['Survived']
    
    # encode data
    X['Sex'] = X['Sex'].replace({'female':1, 'male':0})     
    
    # drop unused columns
    drop_columns = ['PassengerId', 'Name',
                    'Ticket', 'Cabin', 'Embarked']
    X = X.drop(drop_columns, axis=1)
    
    return X, y


def get_Xy_v3(filename='./data/train.csv'):
    """Data Encoding

    Version 3
    * Pclass and Sex encoded as 1/0
    * Age, Fare, SibSp, Parch
    * family_size, is_cabin_notnull, is_large_family
    * is_child, is_boy, is_sibsp_zero, is_parch_zero
    """

    # read data
    all_data = pd.read_csv('./data/train.csv')
    X = all_data.drop('Survived', axis=1)
    y = all_data['Survived']

    # encode data
    X['Sex'] = X['Sex'].replace({'female':1, 'male':0})
    X['family_size'] = X['SibSp'] + X['Parch'] + 1
    X['is_cabin_notnull'] = X['Cabin'].notnull()
    X['is_large_family'] = (X['family_size'] > 4)
    X['is_sibsp_zero'] = (X['SibSp'] == 0)
    X['is_parch_zero'] = (X['Parch'] == 0)

    # comparison with null is false
    # so is_child and is_boy are false when age is null
    X['is_child'] = (X['Age'] < 18)
    X['is_boy'] = (X['Age'] < 18) & (X['Sex'] == 0)

    # drop unused fields
    drop_columns = ['PassengerId', 'Name', 
                    'Ticket', 'Embarked', 'Cabin']
    X = X.drop(drop_columns, axis=1)
    
    return X, y


def get_Xy_v4(filename='./data/train.csv'):
    """Data Encoding

    Version 4
    * Pclass and Sex encoded as 1/0
    * Age, Fare, SibSp, Parch
    * family_size, is_cabin_notnull, is_large_family
    * is_child, is_boy, is_sibsp_zero, is_parch_zero
    * extract Title and dummy encode it
    * dummy encode Embarked
    """

    def extract_title(x):
        title = x.split(',')[1].split('.')[0].strip()
        if title not in ['Mr', 'Miss', 'Mrs', 'Master']:
            title = 'Other'
        return title
    
    # read data
    all_data = pd.read_csv('./data/train.csv')
    X = all_data.drop('Survived', axis=1)
    y = all_data['Survived']

    # encode data
    X['Sex'] = X['Sex'].replace({'female': 1, 'male': 0})
    X['family_size'] = X['SibSp'] + X['Parch'] + 1
    X['is_cabin_notnull'] = X['Cabin'].notnull()
    X['is_large_family'] = (X['family_size'] > 4)
    X['is_sibsp_zero'] = (X['SibSp'] == 0)
    X['is_parch_zero'] = (X['Parch'] == 0)

    # comparison with null is false
    # so is_child and is_boy are false when age is null
    X['is_child'] = (X['Age'] < 18)
    X['is_boy'] = (X['Age'] < 18) & (X['Sex'] == 0)

    # dummy encode title and Embarked
    title = X['Name'].apply(extract_title)
    dummy_title = pd.get_dummies(title, prefix='Title')
    dummy_embarked = pd.get_dummies(X['Embarked'], prefix='Port')
    X = pd.concat([X, dummy_embarked, dummy_title], axis=1)

    # drop unused columns
    drop_columns = ['PassengerId', 'Name',
                    'Ticket', 'Embarked', 'Cabin']
    X = X.drop(drop_columns, axis=1)
    
    return X, y


def get_Xy_v5(filename='./data/train.csv'):
    """Data Encoding

    Version 5 -- Reduced set of features
    * Pclass and Sex encoded as 1/0
    * Age, Fare
    * extract Title and dummy encode it
    * dummy encode Embarked
    """

    def extract_title(x):
        title = x.split(',')[1].split('.')[0].strip()
        if title not in ['Mr', 'Miss', 'Mrs', 'Master']:
            title = 'Other'
        return title
    
    # read data
    all_data = pd.read_csv('./data/train.csv')
    X = all_data.drop('Survived', axis=1)
    y = all_data['Survived']

    # encode data
    X['Sex'] = X['Sex'].replace({'female':1, 'male':0})
    X['family_size'] = X['SibSp'] + X['Parch'] + 1
    X['is_cabin_notnull'] = X['Cabin'].notnull()

    # dummy encode title and Embarked
    title = X['Name'].apply(extract_title)
    dummy_title = pd.get_dummies(title, prefix='Title')
    dummy_embarked = pd.get_dummies(X['Embarked'], prefix='Port')
    X = pd.concat([X, dummy_embarked, dummy_title], axis=1)

    # drop unused columns
    drop_columns = ['PassengerId', 'Name', 'SibSp', 'Parch',
                    'Ticket', 'Embarked', 'Cabin']
    X = X.drop(drop_columns, axis=1)
    
    return X, y


def get_Xy_v6(filename='./data/train.csv'):
    """Data Encoding

    Version 5
    * same as version 4 except encode 3rd class as the number 4
    * to better reflect the added difficultly of being in 3rd class
    """

    def extract_title(x):
        title = x.split(',')[1].split('.')[0].strip()
        if title not in ['Mr', 'Miss', 'Mrs', 'Master']:
            title = 'Other'
        return title

    # read data
    all_data = pd.read_csv('./data/train.csv')
    X = all_data.drop('Survived', axis=1)
    y = all_data['Survived']

    # encode data
    X['Sex'] = X['Sex'].replace({'female': 1, 'male': 0})
    X['Pclass'] = X['Pclass'].replace({1:1, 2:2, 3:4})
    X['family_size'] = X['SibSp'] + X['Parch'] + 1
    X['is_cabin_notnull'] = X['Cabin'].notnull()

    # dummy encode title and Embarked
    title = X['Name'].apply(extract_title)
    dummy_title = pd.get_dummies(title, prefix='Title')
    dummy_embarked = pd.get_dummies(X['Embarked'], prefix='Port')
    X = pd.concat([X, dummy_embarked, dummy_title], axis=1)

    # drop unused columns
    drop_columns = ['PassengerId', 'Name', 'SibSp', 'Parch',
                    'Ticket', 'Embarked', 'Cabin']
    X = X.drop(drop_columns, axis=1)

    return X, y

def get_ct_bycolumn(cols):
    """Column Transform for Features

    """

    ii = WrappedIterativeImputer('Age')

    # Pipelines
    ii_pipe = Pipeline([('ii', ii)])

    # Columns to act on
    ii_cols = ['Pclass', 'Sex', 'Age', 'Title_Master',
                  'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Other']

    if 'Age' in cols:
        # cols.remove is an inplace operation
        # which operates on a reference
        cols = cols.copy()
        cols.remove('Age')
        transformers = [('ii_tr', ii_pipe, ii_cols),
                        ('as_is', 'passthrough', cols)]
        return_cols = ['Age'] + cols
    else:
        transformers = [('as_is', 'passthrough', cols)]
        return_cols = cols

    ct = ColumnTransformer(transformers=transformers)

    return return_cols, ct
