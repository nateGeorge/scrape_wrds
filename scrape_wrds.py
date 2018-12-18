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

hdf_settings_table = {'key': 'data',
                    'mode': 'a',
                    'append': True,
                    'format': 'table',
                    'complib': 'blosc',
                    'complevel': 9}

secd_cols_to_use = ['ajexdi',  # Adjusted Price = (PRCCD / AJEXDI ); “Understanding the Data” on page 91 and on (chapter 6)
                    'cshoc',  # shares outstanding
                    'cshtrd',  # volume
                    'curcdd',
                    'datadate',
                    'eps',
                    'gvkey',
                    'iid',
                    'prccd',  # close
                    'prchd',  # high
                    'prcld',  # low
                    'prcod']  # open

secd_cols = ','.join(secd_cols_to_use)


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
        if (nrows - 1) == current_rows and table == 'sec_shortint':
            print('off by one row, but probly up to date on sec_shortint')
            return True, nrows

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
        return

    df = db.get_table(library=library, table=table)
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
    names_ix -- list of indexes with info (like gvkeyx)
    sec_shortint -- monthly short data

    Keyword arguments:
    db -- connection to wrds db, from make_db_connection() function
    """
    short_tables = ['idxcst_his', 'security', 'names_ix', 'sec_shortint']
    for t in short_tables:
        print(t)
        download_small_table(db=db, table=t)


def load_secd(clean=False):
    secd_filename = FILEPATH + 'hdf/secd.hdf'
    current_df = pd.read_hdf(secd_filename)
    # need to drop messed up values if clean==True
    if clean:
        # drops lots of columns that shouldn't be dropped
        # current_df.dropna(inplace=True)
        # can use to check which entries are bad
        # current_df[current_df['datadate'] >  pd.Timestamp.now(tz='US/Eastern')]
        current_df.drop(current_df[current_df['datadate'] > '11/30/2018'].index, inplace=True)
        # takes about 6 minutes to save
        current_df.to_hdf(secd_filename, **hdf_settings_table)

    return current_df


def download_common_stock_price_history(db, update=True, table='secd', library='comp'):
    """
    downloads data for all common stocks (US and ADR, or tpci column is 0 or F)

    if update=True, will get latest date in current df, then get everything after that
    and add to current df
    """
    # filename from first iteration
    # secd_filename = FILEPATH + 'hdf/common_us_stocks_daily_9-12-2018.hdf'
    latest_date_filename = FILEPATH + 'latest_secd_datadate.txt'
    if not os.path.exists(latest_date_filename):
        # as of 12-13-2018, takes about 40gb to load full thing, around 32 to load datadate
        # this is just for saving the latest date the first time
        current_df = load_secd()
        latest_date = current_df['datadate'].max().strftime('%Y-%m-%d')#.strftime('%m/%d/%y')
        with open(latest_date_filename, 'w') as  f:
            f.write(latest_date)
    else:
        with open(latest_date_filename, 'r') as f:
            latest_date = f.read()

    if latest_date == pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%d'):
        print('up to date on secd')
        return

    # get gvkeys for tpci 0 or F
    # ends up with very slow sql query; avoid
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')
    common_securities = securities[securities['tpci'].isin(['0', 'F'])]
    # # make string for SQL query: WHERE IN
    # # should be like ('item 1', 'item 2', 'item 3')
    # gvkeys_str = '(' + ', '.join(["'" + s + "'" for s in common_securities['gvkey']]) + ')'

    # if you want to count how many rows are there, use this.
    # full data query only took a few seconds even with 1M rows
    # query_str = 'select count(gvkey) from comp.secd where datadate > \'{}\';'.format(latest_date)
    # db.raw_sql(query_str)

    todays_date = pd.Timestamp.now(tz='US/Eastern')
    query_str = 'select {} from {}.{} WHERE datadate > \'{}\' AND datadate <= \'{}\';'# and gvkey IN {};'
    df = db.raw_sql(query_str.format(secd_cols, library, table, latest_date, todays_date), date_cols=['datadate'])
    # drop columns which seem to have weird dates -- don't need with date filter
    # df.drop(df[df['prccd'].apply(lambda x: x is None)].index, inplace=True)
    if not df.shape[0] > 0:
        print("no data to be found!")
        return

    # convert datadate to datetime64
    df['datadate'] = pd.to_datetime(df['datadate']).dt.tz_localize('US/Eastern')
    # colculate market cap
    df['market_cap'] = df['cshoc'] * df['prccd']

    # TODO: create file for storing all updated data and append
    # used once to write data
    # df.to_hdf(FILEPATH + 'hdf/secd_full_9-11-2018_thru_11-30-2018.hdf', **hdf_settings)
    # df.to_hdf(FILEPATH + 'hdf/secd_all_9-11-2018_onward.hdf', **hdf_settings_table)

    # only keep common stocks (tpci = 0 and F)
    common_securities_short = common_securities[['gvkey', 'iid']]
    common_df = df.merge(common_securities_short, on=['gvkey', 'iid'])
    common_df.drop('curcdd', inplace=True, axis=1)  # drop currency column
    # write existing data as hdf table -- first time only
    # current_df.to_hdf(secd_filename, **hdf_settings_table)

    # appends to hdf store
    common_df.to_hdf(secd_filename, **hdf_settings_table)
    # update latest date file
    latest_date = current_df['datadate'].max().strftime('%Y-%m-%d')#.strftime('%m/%d/%y')
    with open(latest_date_filename, 'w') as  f:
        f.write(latest_date)

    del current_df
    del securities
    del df
    del common_df
    del common_securities
    gc.collect()


def hourly_update_check(db):
    """
    checks for updated data once per hour
    """
    while True:
        update_small_tables(db)
        download_common_stock_price_history(db)
        time.sleep(3600)


if __name__ == "__main__":
    db = make_db_connection()
    # update_small_tables(db)
    hourly_update_check(db)
