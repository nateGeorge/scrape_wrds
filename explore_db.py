"""
names_ix has index gvkeyx and index name
- comes from idx_ann table in compd library -- need to rewrite query
idxcst_his has the historical index constituents
"""


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


def list_libs_tables():
    """
    some exploration of the db
    lists libraries, and within each library you can list tables
    """
    db.list_libraries()
    db.list_tables('zacks')
    db.list_tables('ciq')  # don't have permission??
    db.list_tables('comp_global_daily')

    db.list_tables('comp')


def download_entire_table(tablename, library='comp'):
    """
    downloads an entire table by name; library also required.

    default library is the compstat lib
    download entire table
    e.g. tablename='sec_shortint'

    tables downloaded 9-12:
    sec_shortint
    security
    secd


    secd is about 39GB in a pandas df...

    TODO: get latest date already downloaded and use sql query to gegt updates;
    then save to HDF5

    for tables like 'security', check if any more rows and grab new stuff,
    or just grab whole table if cant figure out what new stuff is
    """
    nrows = db.get_row_count(library, tablename)
    print('number of rows:', nrows)
    #db.describe_table(library, tablename)
    # nrows = 1000000
    if tablename == 'secd':  # this is the securities data db -- has historical price data for securities
        cols_to_use = ['ajexdi', # Adjusted Price = (PRCCD / AJEXDI ); “Understanding the Data” on page 91 and on (chapter 6)
                         'cshoc',  # shares outstanding
                         'cshtrd', # volume
                         'datadate',
                         'eps',
                         'gvkey',
                         'iid',
                         'prccd',  # close
                         'prchd',  # high
                         'prcld',  # low
                         'prcod',  # open
                         'tic'  # ticker symbol
                         ]
        df = db.get_table(library, tablename, columns=cols_to_use, obs=nrows)
        df.to_hdf(FILEPATH + 'hdf/{}.hdf'.format(tablename + '_min'), **hdf_settings)
    elif tablename == 'sec_dprc':
        # need to dl in chunks because it is too huge -- expect it to be about 100GB in memory
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

        # WARNING: does not appear to work properly.  probably a sql ordering issue or something
        nobs = 10000000
        for i, start in enumerate(range(0, nrows, nobs), 1):
            print('on part', str(i))
            df = db.get_table(library, tablename, columns=cols_to_use, obs=nobs, offset=start)
            df.to_hdf(FILEPATH + 'hdf/{}.hdf'.format(tablename + '_min_part_' + str(i)), **hdf_settings)
            del df
            gc.collect()
    elif tablename == 'idxcst_his':
        download_index_constituents()
    else:
        print('not one of predefined tables to download')


def check_if_up_to_date(db, df_filepath, table, library='comp'):
    """
    checks if current rows is less than rows in db; returns True is up to date; False if not
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


def download_index_constituents(db, nrows=None, update=False):
    """
    obsolete for now; use download_small_table function instead

    gets historical index constituents from compustat table
    checks if size of table has changed; if so, downloads anew
    TODO: if update is True, then will try to find existing dataframe and only update it
    """
    library = 'comp'
    table = 'idxcst_his'
    # check if any new rows
    df_filepath = FILEPATH + 'hdf/idxcst_his.hdf'
    up_to_date, nrows = check_if_up_to_date(db, df_filepath, table=table, library=library)
    if up_to_date:
        return

    offset = 0
    # turns out the db is fast to download because this is a small table...
    # no need to update row by row, and can't get it working anyhow
    # need to figure out where new things are added, but do later
    # if update:
    #     cst_filepath = FILEPATH + 'hdf/idxcst_his.hdf'
    #     if os.path.exists(cst_filepath):
    #         const_df = pd.read_hdf(cst_filepath)
    #         last_entry = const_df.iloc[-1]
    #         # get new rows plus the last one to check it's the same as the
    #         # last one currently in the hdf file
    #         rows_to_get = nrows - const_df.shape[0] + 1
    #         offset = const_df.shape[0] - 1
    #     else:
    #         rows_to_get = nrows
    #         offset = 0

    df = db.get_table(library=library, table=table, obs=nrows, offset=offset)
    # converts date columns to datetime
    df['from'] = pd.to_datetime(df['from'], utc=True)
    df['thru'] = pd.to_datetime(df['thru'], utc=True)
    df['from'] = df['from'].dt.tz_convert('US/Eastern')
    df['thru'] = df['thru'].dt.tz_convert('US/Eastern')

    df.to_hdf(df_filepath, **hdf_settings)


def download_small_table(db, table, library='comp'):
    """
    downloads table if needs updating

    table can be a tablename in the library; common ones for compustat (comp) are:
    security
    names_ix
    idxcst_his

    .h5 files have same name as table
    """
    df_filepath = FILEPATH + 'hdf/{}.hdf'.format(table)
    up_to_date, nrows = check_if_up_to_date(db, df_filepath, table=table, library=library)
    if up_to_date:
        return

    df = db.get_table(library=library, table=table, obs=nrows)
    if table == 'idxcst_his':
        # converts date columns to datetime
        df['from'] = pd.to_datetime(df['from'], utc=True)
        df['thru'] = pd.to_datetime(df['thru'], utc=True)
        df['from'] = df['from'].dt.tz_convert('US/Eastern')
        df['thru'] = df['thru'].dt.tz_convert('US/Eastern')


    df.to_hdf(df_filepath, **hdf_settings)

    del df
    gc.collect()


def download_common_stock_price_history(db, update=True, table='secd', library='comp'):
    """
    downloads data for all common stocks (US and ADR, or tpci column is 0 or F)

    if update=True, will get latest date in current df, then get everything after that
    and add to current df
    """
    # filename from first iteration
    # secd_filename = FILEPATH + 'hdf/common_us_stocks_daily_9-12-2018.hdf'
    secd_filename = FILEPATH + 'hdf/secd.hdf'
    current_df = pd.read_hdf(secd_filename)
    latest_date = current_df['datadate'].max().strftime('%m/%d/%y')

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

    query_str = 'select {} from {}.{} WHERE datadate > \'{}\';'# and gvkey IN {};'
    df = db.raw_sql(query_str.format(secd_cols, library, table, latest_date), date_cols=['datadate'])
    # drop columns which seem to have weird dates
    df.drop(df['prccd'].apply(lambda x: x is None).index, inplace=True)
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
    df.to_hdf(FILEPATH + 'hdf/secd_all_9-11-2018_onward.hdf', **hdf_settings_table)

    # only keep common stocks (tpci = 0 and F)
    common_securities_short = common_securities[['gvkey', 'iid']]
    common_df = df.merge(common_securities_short, on=['gvkey', 'iid'])
    common_df.drop('curcdd', inplace=True, axis=1)  # drop currency column
    # write existing data as hdf table -- first time only
    # current_df.to_hdf(secd_filename, **hdf_settings_table)

    # appends to hdf store
    common_df.to_hdf(secd_filename, **hdf_settings_table)

    del current_df
    del securities
    del df
    del common_df
    del common_securities
    gc.collect()


def get_stock_hist_df(gvkey, library='comp', tablename='secd'):
    df = db.raw_sql('select {} from {}.{} WHERE gvkey = {};'.format(secd_cols, library, tablename, gvkey), date_cols=['datadate'])
    return df


def download_all_security_data():
    """
    downloads full security data history for sp600
    I think this was actually used to get all historical stock data actually,
    not just sp600.

    TODO: get latest date and download updates
    """
    df = pd.read_hdf(FILEPATH + 'hdf/idxcst_his.hdf')

    sp600_df = df[df['gvkeyx'] == '030824']
    sp600_gvkeys = np.unique(sp600_df['gvkey'].values)
    sp600_gvkeys_strings = ["'" + gv + "'" for gv in sp600_gvkeys]
    sp600_gvkeys_string = ', '.join(sp600_gvkeys_strings)

    # reads in all securities
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')
    all_gvkeys = securities['gvkey'].values

    remaining_gvs = list(set(all_gvkeys).difference(set(sp600_gvkeys)))

    # raw sql to get historical security data
    # goes through all securities and downloads historical price data
    # chunk through remaining gvkeys in 10 chunks
    chunk_size = len(remaining_gvs) // 10
    for i, ch in enumerate(range(0, len(remaining_gvs) + 1, chunk_size)):
        # first make strings out of gvkeys for SQL query
        start = ch
        if ch + chunk_size > len(remaining_gvs):
            gvkeys_strings = ["'" + gv + "'" for gv in remaining_gvs[start:]]
        else:
            gvkeys_strings = ["'" + gv + "'" for gv in remaining_gvs[start:ch + chunk_size]]

        start = time.time()
        jobs = []
        # 10 threads per cpu for 8 cores; default is 5 per CPU
        # seems like 5 simultaneous queries is max -- run in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            for gv in gvkeys_strings:
                jobs.append((gv, executor.submit(get_stock_hist_df, gv)))

        dfs = []
        for gv, j in jobs:
            # print(gv)
            dfs.append(j.result())

        end = time.time()
        print('took', int(end - start), 'seconds')

        big_df = pd.concat(dfs)
        big_df['datadate'] = pd.to_datetime(big_df['datadate']).dt.tz_localize('US/Eastern')
        # big_df['datadate'] = pd.Timestamp(big_df['datadate'])  # doesn't work!!
        # big_df['datadate'].dt.tz_localize('US/Eastern')
        # TODO: dynamically set date instead of hard copy
        big_df.to_hdf(FILEPATH + 'hdf/daily_security_data__chunk_{}_9-15-2018.hdf'.format(str(i)), **hdf_settings)
        del jobs
        del dfs
        del big_df
        gc.collect()

        # 30 seconds per 50 -- should take about 20m for 2k
        # took 1282s for 2127 gvkeys


def load_and_combine_sec_dprc():
    """
    loads all security data from sec_dprc table
    """
    dfs = []
    for i in tqdm(range(1, 13)):
        # print(i)
        dfs.append(pd.read_hdf(FILEPATH + 'hdf/sec_dprc_min_part_{}.hdf'.format(str(i))))

    df = pd.concat(dfs)

    # get only common stocks
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')
    # gvkeys = df['gvkey'].unique()
    # I think 0 or F for tpci are common or ADR, which are stocks you can buy
    common_stocks = securities[securities['tpci'].isin(['0', 'F'])]
    common_stocks.drop(common_stocks[common_stocks['ibtic'].isnull()].index, inplace=True)  # these seem to be weird tickers; buyouts or something
    # ignore stocks on canadian exchanges
    common_stocks.drop(common_stocks[common_stocks['iid'].str.contains('C')].index, inplace=True)
    # check to make sure only one iid per gvkey -- not quite
    gvkey_grp = common_stocks.groupby('gvkey')
    num_iids = gvkey_grp['iid'].nunique()
    num_iids.mean()
    num_iids[num_iids > 1]

    common_df = df[df['gvkey'].isin(set(common_stocks['gvkey'].unique()))]
    common_df = common_df[common_df['iid'].isin(set(common_stocks['iid'].unique()))]

    # don't use CAD stocks
    common_df.drop(common_df[common_df['curcdd'] == 'CAD'].index, inplace=True)
    # no longer need currency, all USD
    common_df.drop('curcdd', axis=1, inplace=True)

    common_df['datadate'] = pd.to_datetime(common_df['datadate']).dt.tz_localize('US/Eastern')
    common_df['market_cap'] = common_df['cshoc'] * common_df['prccd']
    common_df.to_hdf(FILEPATH + 'hdf/common_us_stocks_daily_9-12-2018.hdf', **hdf_settings)

    # add ticker and remove iid and gvkey -- should just merge or something
    # for gvkey in tqdm(common_df['gvkey'].unique()):
    #     common_df.at[common_df['gvkey'] == gvkey, 'ticker'] = securities[securities['gvkey'] == gvkey]['tic']



def get_historical_constituents_wrds_hdf(date_range=None, index='S&P Smallcap 600 Index'):
    # adapted from beat_market_analysis constituent_utils.py
    """
    gets historical constituents from WRDS file

    common indexes as represented in the idx_ann table:
    SP600: S&P Smallcap 600 Index
    SP400: S&P Midcap 400 Index
    SP500: S&P 500 Comp-Ltd (there's another one with Wed instead of Ltd which I don't know what it is)
    SP1500: S&P 1500 Super Composite

    NASDAQ 100: Nasdaq 100
    """
    idx_df = pd.read_hdf(FILEPATH + 'hdf/names_ix.hdf')
    gvkeyx = idx_df[idx_df['conm'] == index]['gvkeyx'].values
    if len(gvkeyx) > 1:
        print('more than 1 gvkeyx, exiting:')
        print(idx_df[idx_df['conm'] == index])
        return

    gvkeyx = gvkeyx[0]

    # TODO: get latest file
    # parse dates not working for hdf, parse_dates=['from', 'thru'], infer_datetime_format=True)
    const_df = pd.read_hdf(FILEPATH + 'hdf/idxcst_his.hdf')
    # only need to do this once, then after it's saved, good to go
    # TODO: put this in a clean function and do when saving the file
    # df['from'] = pd.to_datetime(df['from'], utc=True)
    # df['thru'] = pd.to_datetime(df['thru'], utc=True)
    # df['from'] = df['from'].dt.tz_convert('US/Eastern')
    # df['thru'] = df['thru'].dt.tz_convert('US/Eastern')
    # df.to_hdf(FILEPATH + 'hdf/index_constituents_9-12-2018.hdf', **hdf_settings)

    # need to join up with other dataframe maybe, for now, just use gvkeyx which is
    # 030824 for sp600
    # df2 = pd.read_hdf(FILEPATH + 'hdf/names_ix.hdf')

    single_idx_df = const_df[const_df['gvkeyx'] == gvkeyx].copy()
    # combine with securities for ticker symbol
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')
    # abbreviated securities df; only ticker, gvkey, and iid
    sec_short = securities[['tic', 'gvkey', 'iid']]

    single_idx_df = single_idx_df.merge(sec_short, on=['gvkey', 'iid'])

    # get stocks' gvkeys for sql search -- no longer needed
    # gvkeys = single_idx_df['gvkey'].values

    # create dataframe with list of constituents for each day
    start = single_idx_df['from'].min()
    # get todays date and reset hour, min, sec to 0s
    # TODO: if not latest date; use date of datafile as latest
    end = pd.Timestamp.today(tz='US/Eastern').replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None).tz_localize('US/Eastern')

    # replace NaT with tomorrow's date
    # gives copy warning but can't get rid of it...
    single_idx_df['thru'].fillna(end + pd.DateOffset(days=1), inplace=True)


    nyse = mcal.get_calendar('NYSE')
    # gets all dates
    # date_range = mcal.date_range(start=start, end=end)
    # gets only dates valid for NYSE -- doesn't seem to match historical data
    if date_range is None:
        date_range = nyse.valid_days(start_date=start.date(), end_date=end.date()).tz_convert('US/Eastern')
    else:
        # cutoff at earliest date for index
        date_range = np.array(sorted(date_range))
        date_range = date_range[date_range >= start]


    constituent_companies = OrderedDict()
    # constituent_tickers = OrderedDict()
    lengths = []

    # TODO: multiprocessing to speed up
    # takes about 10s for nasdaq 100
    for d in tqdm(date_range):
        # if date is within stock's from and thru, add to list
        # stocks were removed on 'thru', so if it is the 'thru' date, then shouldn't be included
        # but stocks were added on 'from' date, so include stocks on 'from' date
        # use dataframe masking
        date_string = d.strftime('%Y-%m-%d')
        current_stocks = single_idx_df[(single_idx_df['from'] <= d) & (single_idx_df['thru'] > d)]
        current_companies = current_stocks[['gvkey', 'iid']]  # company names
        # current_tickers = current_stocks['co_tic']  # company tickers
        constituent_companies[date_string] = current_companies
        # constituent_tickers[date_string] = current_tickers
        lengths.append(current_companies.shape[0])

    # look at number of constituents as a histogram; mostly 600 but a few above and below
    # pd.value_counts(lengths)
    # plt.hist(lengths)
    # plt.show()

    # TODO:
    # need to check that no tickers are used for multiple companies

    # get unique dates where changes were made
    unique_dates = set(single_idx_df['from'].unique()) | set(single_idx_df['thru'].unique())

    return constituent_companies, unique_dates


def spy_20_smallest():
    """
    tries to implement 20 smallest SPY strategy from paper (see beat_market_analysis github repo)
    """
    # merge historical constituents for sp600 with daily price, eps, and market cap data
    # see what returns are on yearly rebalance for 20 smallest marketcap stocks
    # just get first of year dates, then get company market caps
    # get smallest 20 market caps, get close price
    # get close price a year later, calculate overall return
    # repeat ad nauseum

    # common_stocks = pd.read_hdf(FILEPATH + 'hdf/common_us_stocks_daily_9-12-2018.hdf')
    sp600_stocks = pd.read_hdf(FILEPATH + 'hdf/sp600_daily_security_data_9-15-2018.hdf')
    sp600_stocks['market_cap'] = sp600_stocks['cshoc'] * sp600_stocks['prccd']
    # sp600 index data starts in 1994
    years = sp600_stocks['datadate'][sp600_stocks['datadate'].dt.year >= 1994].dt.year.unique()

    first_days = []
    sp600_dates = sorted(sp600_stocks['datadate'].unique())
    constituent_companies, unique_dates = get_historical_constituents_wrds_hdf(sp600_dates)

    for y in tqdm(years[1:]):  # first year starts on sept
        year_dates = [d for d in sp600_dates if d.year == y]
        first_days.append(min(year_dates))

    # '1998-01-02' giving key error in constituent_companies
    price_chg_1y = OrderedDict()
    smallest_20 = OrderedDict()
    smallest_20_1y_chg = OrderedDict()

    # TODO: get latest price if stopped trading during the year; figure out mergers/buyouts, etc
    # TODO: get tickers
    for start, end in tqdm(zip(first_days[4:-1], first_days[5:])):  # 2000 onward is [5:] ; market cap not available until 1999 for these stocks
        datestr = start.strftime('%Y-%m-%d')
        constituents = constituent_companies[datestr]
        current_daily_data = sp600_stocks[sp600_stocks['datadate'] == start]
        one_year_daily_data = sp600_stocks[sp600_stocks['datadate'] == end]
        # TODO: figure out why a few hundred are missing in the daily data from the constituent list
        # AIR ('001004') is not in common_stocks, figure out why
        full_const = constituents.merge(current_daily_data, on=['gvkey', 'iid'])
        full_const_1y = constituents.merge(one_year_daily_data, on=['gvkey', 'iid'])
        # get adjusted closes for constituents now and 1y in future
        const_current_price = full_const[['gvkey', 'iid', 'ajexdi', 'prccd']]
        const_future_price = full_const_1y[['gvkey', 'iid', 'ajexdi', 'prccd']]
        const_current_price['adj_close'] = const_current_price['prccd'] / const_current_price['ajexdi']
        const_future_price['adj_close_1y_future'] = const_future_price['prccd'] / const_future_price['ajexdi']
        const_current_price.drop(['prccd', 'ajexdi'], inplace=True, axis=1)
        const_future_price.drop(['prccd', 'ajexdi'], inplace=True, axis=1)
        # get % price change for each
        const_price_change = const_current_price.merge(const_future_price, on=['gvkey', 'iid']).drop_duplicates()
        const_price_change['1y_pct_chg'] = (const_price_change['adj_close_1y_future'] - const_price_change['adj_close']) / const_price_change['adj_close']
        price_chg_1y[datestr] = const_price_change

        bottom_20 = full_const.sort_values(by='market_cap', ascending=True).iloc[:20]
        smallest_20[datestr] = bottom_20
        bottom_20_price_chg = const_price_change[const_price_change['gvkey'].isin(set(bottom_20['gvkey']))]
        bottom_20_price_chg.reset_index(inplace=True, drop=True)
        if bottom_20_price_chg.shape[0] == 0:  # everything was acquired/bankrupt, etc, like in 2006 and 07 I think
            last_idx = 0
        else:
            last_idx = bottom_20_price_chg.index[-1]

        # get stocks missing from price changes, and use last price to get price change
        missing_gvkeys = list(set(bottom_20['gvkey']).difference(set(bottom_20_price_chg['gvkey'])))
        for m in missing_gvkeys:
            last_idx += 1  # make an index for creating dataframe with last price, so we can append it to the bottom_20_price_chg df
            price_chg_dict = {}
            iid = bottom_20[bottom_20['gvkey'] == m]['iid'].values
            if len(iid) > 1:
                print('shit, iid length >1')
            iid = iid[0]
            last_data = sp600_stocks[(sp600_stocks['gvkey'] == m) & (sp600_stocks['iid'] == iid)][['prccd', 'ajexdi']].dropna().iloc[-1]
            last_price = last_data['prccd'] / last_data['ajexdi']
            price_chg_dict['gvkey'] = m
            price_chg_dict['iid'] = iid
            # TODO: check this isn't more than one result, may need to filter by iid too
            price_chg_dict['adj_close'] = const_current_price[const_current_price['gvkey'] == m]['adj_close'].values[0]
            price_chg_dict['adj_close_1y_future'] = last_price
            price_chg_dict['1y_pct_chg'] = (last_price - price_chg_dict['adj_close']) / price_chg_dict['adj_close']
            bottom_20_price_chg = bottom_20_price_chg.append(pd.DataFrame(price_chg_dict, index=[last_idx])[bottom_20_price_chg.columns.tolist()])  # TODO: check if append works with out-of-order columns


        # TODO: find next stock in bottom 20 at time the other was put out, and see how it does

        smallest_20_1y_chg[datestr] = bottom_20_price_chg
        price_chg_1y[datestr] = bottom_20_price_chg['1y_pct_chg'].sum() / 20  # assume others not in here are 0 for now
        # get the overall price changes each year


    annualized_return = (np.prod([1 + p for p in price_chg_1y.values()]) ** (1/len(price_chg_1y.values())) - 1) * 100
    plt.plot(price_chg_1y.keys(), price_chg_1y.values())
    plt.scatter(price_chg_1y.keys(), price_chg_1y.values())
    plt.xticks(rotation=90)
    plt.title('bottom 20 SP600 stocks yearly returns, annualized return = ' + str(round(annualized_return, 1)))
    plt.ylabel('% return per year')
    plt.tight_layout()
    plt.show()

    # to get tickers
    smallest_20_1y_chg['2017-01-03'].merge(securities[['gvkey', 'iid', 'tic']], on=['gvkey', 'iid'])
    bottom_20.merge(securities[['gvkey', 'iid', 'tic']], on=['gvkey', 'iid'])


    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')

    bottom_20_tickers = bottom_20.merge(securities, on=['gvkey', 'iid'])


    # TODO: deal with acquisitions: dlrsni 01 is acquired, 02 is bankrupt, 03 is liquidated
    # https://wrds-web.wharton.upenn.edu/wrds/support/Data/_001Manuals%20and%20Overviews/_001Compustat/_001North%20America%20-%20Global%20-%20Bank/_000dataguide/index.cfm

    # get gvkeys missing in price changes and check for bankruptcy or acquisitions, etc
    missing_gvkeys = list(set(bottom_20['gvkey']).difference(set(bottom_20_price_chg['gvkey'])))
    missing = bottom_20[bottom_20['gvkey'].isin(missing_gvkeys)]
    missing_merged = missing.merge(securities[['gvkey', 'iid', 'dlrsni', 'tic']])
    missing_merged[['tic', 'dlrsni']]
    securities[securities['gvkey'] == '010565']


    # TODO: is it different/better to rebalance on a certain day/month?


def secd_info():
    """
    info of first 1M rows of secd:

    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 41 columns):
    gvkey             1000000 non-null object
    iid               1000000 non-null object
    datadate          1000000 non-null object
    tic               1000000 non-null object
    cusip             1000000 non-null object
    conm              1000000 non-null object
    curcddv           5861 non-null object
    capgn             29 non-null float64
    cheqv             85 non-null float64
    div               5780 non-null float64
    divd              5694 non-null float64
    divdpaydateind    0 non-null object
    divsp             129 non-null float64
    dvrated           2875 non-null float64
    paydateind        2 non-null object
    anncdate          2776 non-null object
    capgnpaydate      29 non-null object
    cheqvpaydate      82 non-null object
    divdpaydate       5691 non-null object
    divsppaydate      128 non-null object
    paydate           5772 non-null object
    recorddate        2906 non-null object
    curcdd            999696 non-null object
    adrrc             4202 non-null float64
    ajexdi            999696 non-null float64
    cshoc             439670 non-null float64
    cshtrd            999677 non-null float64
    dvi               379938 non-null float64
    eps               309295 non-null float64
    epsmo             309295 non-null float64
    prccd             999696 non-null float64
    prchd             986959 non-null float64
    prcld             985637 non-null float64
    prcod             224624 non-null float64
    prcstd            999696 non-null float64
    trfd              733884 non-null float64
    exchg             1000000 non-null float64
    secstat           1000000 non-null object
    tpci              1000000 non-null object
    cik               922655 non-null object
    fic               1000000 non-null object
    dtypes: float64(20), object(21)
    memory usage: 312.8+ MB

    so we can ignore most of those middle columns

    cols_to_use = ['ajexdi',
                     'cshoc',  # shares outstanding
                     'cshtrd', # volume
                     'datadate',
                     'eps',
                     'prccd',
                     'prchd',
                     'prcld',
                     'prcod',
                     'tic'  # maybe want to get iid too, not sure
                     ]

    other_cols = ['adrrc',
     'anncdate',
     'capgn',
     'capgnpaydate',
     'cheqv',
     'cheqvpaydate',
     'curcdd',
     'curcddv',
     'cusip',
     'datadate',
     'div',
     'divd',
     'divdpaydate',
     'divdpaydateind',
     'divsp',
     'divsppaydate',
     'dvi',
     'dvrated',
     'epsmo',
     'exchg',
     'fic',
     'gvkey',
     'iid',
     'paydate',
     'paydateind',
     'prcstd',
     'recorddate',
     'secstat',
     'tic',
     'tpci',
     'trfd']
    """
    pass


def test_sql_queries():
    pass
    # with limit for testing
    # df = db.raw_sql('select {} from {}.{} WHERE gvkey IN ({}) LIMIT 10;'.format(','.join(cols_to_use), library, tablename, sp600_gvkeys_string), date_cols=['datadate'])
    # takes a really long time...
    # # df = db.raw_sql('select {} from {}.{} WHERE gvkey IN ({});'.format(','.join(cols_to_use), library, tablename, sp600_gvkeys_string), date_cols=['datadate'])

    # see how long one query takes -- about 2s
    # start = time.time()
    # df = db.raw_sql('select {} from {}.{} WHERE gvkey = {};'.format(','.join(cols_to_use), library, tablename, sp600_gvkeys_strings[0]), date_cols=['datadate'])
    # end = time.time()
    # print('took', int(end - start), 'seconds')

    # takes about 2h linearly
    # dfs = []
    # for gv in tqdm(sp600_gvkeys_strings):
    #     df = db.raw_sql('select {} from {}.{} WHERE gvkey = {};'.format(','.join(cols_to_use), library, tablename, gv), date_cols=['datadate'])
    #     dfs.append(df)

    # testing
    # df = db.raw_sql('select {} from {}.{} WHERE gvkey = \'001004\' LIMIT 10;'.format(','.join(cols_to_use), library, tablename), date_cols=['datadate'])


def testing_db():
    """
    looks like some code that tests some db functions and explores them
    """
    df = db.get_table('comp', 'security', obs=10)

    db.get_table('crsp', 'dsf', columns=['cusip', 'permno', 'date', 'bidlo', 'askhi'], obs=100)



    # compustat data

    # short data
    db.get_table('comp', 'sec_shortint', obs=100)

    # quarterly fundamentals
    db.get_table('comp', 'fundq')
    # annual
    db.get_table('comp', 'funda')


    # industry quarterly
    db.get_table('comp', 'aco_indstq')
    # annual
    db.get_table('comp', 'aco_indsta')

    # index prices daily
    db.get_table('comp', 'idx_mth')

    # simplified financial statement extract daily
    db.get_table('comp', 'funda')  # seems to be the same as annual fundamentals

    # annual index fundamentals
    db.get_table('comp', 'idx_ann')

    #  monthly security data
    db.get_table('comp', 'secm', obs=100)

    # index constituents
    db.get_table('comp', 'idxcst_his')


    # market cap/price, daily data
    db.get_table('comp', 'secd', obs=100)


    # OTC pricing
    db.get_table('otc', 'endofday', obs=100)

    # gets acquisition spending; aqcy column
    df4 = db.raw_sql('select * from comp.fundq WHERE gvkey = \'010519\';')


def get_nasdaq_100_constituents():
    """
    gets historical nasdaq 100 constituents
    then looks at
    """
    constituent_companies, unique_dates = get_historical_constituents_wrds_hdf(date_range=None, index='Nasdaq 100')
