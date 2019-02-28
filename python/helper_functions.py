import pandas as pd
import numpy as np
import re
from sqlalchemy.types import SmallInteger, Integer, BigInteger

def to_csv_with_types(df, filename):
    """
    Save df to csv file and save df.dtypes to csv file.

    If filename ends in .gz, Pandas will use gzip compression.

    This is intended to be used after optimizing df column types.
    Read back with: from_csv_with_types()
    """

    filename_types = filename.split('.')[0] + '_types.csv'
    dtypes = df.dtypes.to_frame('dtypes').reset_index()

    dtypes.to_csv(filename_types, index=False)
    df.to_csv(filename, index=False)

def from_csv_with_types(filename, nrows=None):
    """
    Read df.dtypes from csv file and read df from csv file.

    If filename ends in .gz, Pandas will use gzip decompression.
    This is the complement of to_csv_with_types().
    """

    filename_types = filename.split('.')[0] + '_types.csv'

    types = pd.read_csv(filename_types).set_index('index').to_dict()
    dtypes = types['dtypes']

    dates = [key for key, value in dtypes.items() if value.startswith('datetime')]
    for field in dates:
        dtypes.pop(field)

    return pd.read_csv(filename, parse_dates=dates, dtype=dtypes, nrows=nrows)

def optimize_df_dtypes(df, cutoff=0.05):
    """
    Downcasts DataFrame Column Types to fit the data.

    :param df:
    Dataframe to optimize.

    :param cutoff:
    Specifies cutoff ratio of unique values to rows for converting to categories.

    :return:
    Optimized DataFrame.
    """

    df = df.copy()

    # int64 -> smallest uint allowed by data
    df_int = df.select_dtypes(include=[np.int])
    df_int = df_int.apply(pd.to_numeric, downcast='unsigned')
    df[df_int.columns] = df_int

    # object -> category, if less than 5% of values are unique
    df_obj = df.select_dtypes(include=['object'])
    s = df_obj.nunique() / df.shape[0]
    columns = s.index[s <= cutoff].values
    if len(columns) > 0:
        df_cat = df[columns].astype('category')
        df[columns] = df_cat

    return df

def optimize_db_dtypes(df):
    """
    Choose smallest ANSI SQL Column Type that fits the optimized DataFrame.

    Relies on:
    from sqlalchemy.types import SmallInteger, Integer, BigInteger
    """
    small_int = {col: SmallInteger for col in df.select_dtypes(
        include=[np.int16, np.uint16, np.int8, np.uint8]).columns}

    integer = {col: Integer for col in df.select_dtypes(
        include=[np.int32, np.uint32]).columns}

    big_int = {col: BigInteger for col in df.select_dtypes(
        include=[np.int64, np.uint64]).columns}

    dtypes = {**small_int, **integer, **big_int}

    return dtypes

def is_unique(df, cols):
    """Fast determination of multi-column uniqueness."""
    return not (df.duplicated(subset=cols)).any()

def mem_usage(df):
    """Returns a string representing df memory usage in MB."""
    mem = df.memory_usage(deep=True).sum()
    mem = mem / 2 ** 20 # covert to megabytes
    return f'{mem:03.2f} MB'

def is_int(s):
    """Returns True if all non-null values are integers.

    Useful for determining if the df column (pd.Series) is
    float just to hold missing values.
    """
    notnull = s.notnull()
    is_integer = s.apply(lambda x: (x%1 == 0.0))
    return (notnull == is_integer).all()

def convert_camel_case(name):
    """
    CamelCase to snake_case.

    This is from:
    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case#answer-1176023
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


