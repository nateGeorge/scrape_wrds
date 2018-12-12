import os
import gc
import time
import datetime
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas_market_calendars as mcal
import pandas as pd
from tqdm import tqdm
import wrds

FILEPATH = '/home/nate/Dropbox/data/wrds/compustat_north_america/'

hdf_settings = {'key': 'data',
                'mode': 'w',
                'complib': 'blosc',
                'complevel': 9}


def make_db_connection():
    """
    creates connection to WRDS database
    need to enter credentials to log in
    """
    wrds_uname = os.environ.get('wrds_username')
    wrds_pass = os.environ.get('wrds_password')
    # tries to use pgpass file; see here:
    # https://wrds-www.wharton.upenn.edu/pages/support/accessing-wrds-remotely/troubleshooting-pgpass-file-remotely/
    db = wrds.Connection(wrds_username=wrds_uname, wrds_password=wrds_pass)

    # saves credentials, but not pgpass working
    # db.create_pgpass_file()
    return db


def check_if_up_to_date(db, df_filepath, table, library='comp'):
    """
    checks if current rows is less than rows in db on WRDS;
    returns True is up to date; False if not

    Keyword arguments:
    db -- database connection (from make_db_connection)
    df_filepath --
    table -- string, name of table (e.g. names_ix)
    library -- string, name of library (e.g. comp for compustat)
    """
    if os.path.exists(df_filepath):
        current_df = pd.read_hdf(df_filepath)
        current_rows = current_df.shape[0]
    else:
        current_rows = 0

    nrows = db.get_row_count(library=library, table=table)
    if nrows == current_rows:
        print('up to date')
        return True, nrows
    elif nrows < current_rows:
        print('number of available rows is less than number in current db;')
        print('something is wrong...')
        return True, nrows
    else:
        print('db needs updating')
        return False, nrows


def download_small_table(db, table, library='comp', return_table=False):
    """
    downloads table if needs updating
    This is intended for smaller tables, where the entire table can be
    downloaded at once and can overwrite the old table.

    table can be a tablename in the library; common ones for compustat (comp) are:
    security
    names_ix
    idxcst_his

    .h5 files have same name as table

    Keyword arguments:
    db -- database connection (from make_db_connection)
    table -- string, name of table (e.g. names_ix)
    library -- string, name of library (e.g. comp for compustat)
    return_table -- boolean, if True, returns downloaded dataframe
    """
    df_filepath = FILEPATH + 'hdf/{}.hdf'.format(table)
    up_to_date, nrows = check_if_up_to_date(db, df_filepath, table=table, library=library)
    if up_to_date:
        if return_table:
            return None

    df = db.get_table(library=library, table=table, obs=nrows)
    if table == 'idxcst_his':
        # converts date columns to datetime
        df['from'] = pd.to_datetime(df['from'], utc=True)
        df['thru'] = pd.to_datetime(df['thru'], utc=True)
        df['from'] = df['from'].dt.tz_convert('US/Eastern')
        df['thru'] = df['thru'].dt.tz_convert('US/Eastern')


    df.to_hdf(df_filepath, **hdf_settings)
    if return_table:
        return df


def get_stock_hist_df(gvkey, library='comp', tablename='secd'):
    """
    gets historical daily data for a single stock in the db.

    Keyword arguments:
    gvkey -- the number ID for the stock (get from 'security' table)
    """
    cols_to_use = ['ajexdi',
                        'cshoc',
                        'cshtrd',
                        'curcdd',
                        'datadate',
                        'eps',
                        'gvkey',
                        'iid',
                        'prccd',
                        'prchd',
                        'prcld',
                        'prcod']
    df = db.raw_sql('select {} from {}.{} WHERE gvkey = {};'.format(','.join(cols_to_use), library, tablename, gvkey), date_cols=['datadate'])
    return df


def update_small_tables(db):
    """
    small tables:
    idxcst_his -- historical index constituents
    security -- list of securities with gvkey and other info
    names_ix -- list of indexes with info (like gxkey)
    sec_shortint -- monthly short data

    Keyword arguments:
    db -- connection to wrds db, from make_db_connection() function
    """
    short_tables = ['idxcst_his', 'security', 'names_ix', 'sec_shortint']
    for t in short_tables:
        download_small_table(db=db, table=t)



db = make_db_connection()
