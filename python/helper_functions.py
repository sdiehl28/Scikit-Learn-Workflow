import pandas as pd
import numpy as np
import re
from sqlalchemy.types import SmallInteger, Integer, BigInteger
from IPython.display import HTML, display

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


def optimize_df_dtypes(df, ignore=None):
    """
    Downcasts DataFrame Column Types.

    :param df:
    Dataframe to optimize.

    :param cutoff:
    Specifies cutoff ratio of unique values to rows for converting to categories.

    :param ignore
    Specifies which fields to exclude from downcasting.

    :return:
    Optimized DataFrame.
    """

    df = df.copy()

    # columns to consider for downcasting
    process_cols = df.columns
    if ignore and len(ignore) > 0:
        process_cols = df.columns.difference(ignore)

        if len(process_cols) == 0:
            return df

    # get the integer columns, if any
    df_int = df[process_cols].select_dtypes(include=[np.int])

    # if there are some integer columns, downcast them
    if len(df_int.columns) > 0:
        df_int = df_int.apply(pd.to_numeric, downcast='unsigned')
        df[df_int.columns] = df_int

    # automated conversion to categories can be problematic
    # if a category is warranted, probably a CategoryDType should be created

    # get the object columns, if any
    # df_obj = df[process_cols].select_dtypes(include=['object'])
    #
    # # if there are some object columns, convert to category if less than 10% unique
    # if len(df_obj.columns) > 0:
    #     s = df_obj.nunique() / df.shape[0]
    #     columns = s.index[s <= cutoff].values
    #     if len(columns) > 0:
    #         df_cat = df[columns].astype('category')
    #         df[columns] = df_cat

    return df


def optimize_db_dtypes(df):
    """
    Choose smallest ANSI SQL Column Type for integer that fits the optimized DataFrame.

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

def mem_usage(df):
    """Returns a string representing df memory usage in MB."""
    mem = df.memory_usage(deep=True).sum()
    mem = mem / 2 ** 20  # covert to megabytes
    return f'{mem:03.2f} MB'


def is_int(s):
    """Returns True if all non-null values are integers.

    Useful for determining if the df column (pd.Series) is
    float just to hold missing values.
    """
    notnull = s.notnull()
    is_integer = s.apply(lambda x: (x % 1 == 0.0))
    return (notnull == is_integer).all()


def convert_camel_case(name):
    """
    CamelCase to snake_case.

    This is from:
    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case#answer-1176023
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def is_unique(df, cols):
    """Fast determination of multi-column uniqueness."""
    return not (df.duplicated(subset=cols)).any()

def game_id_to_url(game_id):
    home = game_id[:3]
    url = 'https://www.baseball-reference.com/boxes/' + home + '/' + game_id + '.shtml'
    display(HTML(f'<a href="{url}">{game_id}</a>'))

def order_cols(df,cols):
    """Put columns in cols first, followed by rest of columns"""
    rest = [col for col in df.columns if col not in cols]
    df = df[cols + rest]
    return df