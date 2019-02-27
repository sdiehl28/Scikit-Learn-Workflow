from sqlalchemy.types import SmallInteger, Integer, BigInteger

def to_csv_with_types(df, file_prefix, compression=False):
    """Save df to csv and save df.dtypes to csv"""

    dtypes = df.dtypes.to_frame('dtypes').reset_index()
    filename_types = file_prefix + '_types.csv'
    dtypes.to_csv(filename_types, index=False)

    if compression:
        filename = file_prefix + '.csv.gz'
        df.to_csv(filename, compression='gzip', index=False)
    else:
        filename = file_prefix + '.csv'
        df.to_csv(filename, index=False)

def from_csv_with_types(file_prefix, compression=False, nrows=None):
    """Read df.dtypes from csv and read df from csv"""

    filename_types = file_prefix + '_types.csv'
    types = pd.read_csv(filename_types).set_index('index').to_dict()
    dtypes = types['dtypes']

    dates = [key for key, value in dtypes.items() if value.startswith('datetime')]
    for field in dates:
        dtypes.pop(field)

    if compression:
        filename = file_prefix + '.csv.gz'
        df = pd.read_csv(filename, compression='gzip', parse_dates=dates, dtype=dtypes, nrows=nrows)
    else:
        df = pd.read_csv(file_prefix + '.csv', parse_dates=dates, dtype=dtypes, nrows=nrows)
    return df

def is_unique(df, cols):
    # faster than using groupby
    return not (df.duplicated(subset=cols)).any()

def mem_usage(df):
    mem = df.memory_usage(deep=True).sum()
    mem = mem / 2 ** 20 # covert to megabytes
    return f'{mem:03.2f} MB'

def is_all_int(s):
    """Returns True if all non-null values are integers"""
    notnull = s.notnull()
    is_integer = s.apply(lambda x: (x%1 == 0.0))
    return (notnull == is_integer).all()

def optimize_pandas_dtypes(df, cutoff=0.05):
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

def optimize_database_dtypes(df):
    """Chose best DB column type from DataFrame column type"""
    small_int = {col: SmallInteger for col in df.select_dtypes(
        include=[np.int16, np.uint16, np.int8, np.uint8]).columns}

    integer = {col: Integer for col in df.select_dtypes(
        include=[np.int32, np.uint32]).columns}

    big_int = {col: BigInteger for col in df.select_dtypes(
        include=[np.int64, np.uint64]).columns}

    dtype = {**small_int, **integer, **big_int}

    return dtype