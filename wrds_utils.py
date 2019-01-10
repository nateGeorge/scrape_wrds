import os
import platform
import datetime
import pickle as pk
from pytz import timezone
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_market_calendars as mcal

eastern = timezone('US/Eastern')

import pandas as pd

FILEPATH = '/home/nate/Dropbox/data/wrds/compustat_north_america/'


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.

    returns datetime object
    """
    if platform.system() == 'Windows':
        return datetime.datetime.fromtimestamp(os.path.getctime(path_to_file))
    else:
        stat = os.stat(path_to_file)
        try:
            return datetime.datetime.fromtimestamp(stat.st_birthtime)
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return datetime.datetime.fromtimestamp(stat.st_mtime)


def load_small_table(table):
    """
    Keyword arguments:
    table -- string with tablename (e.g. names_ix)
    """
    df_filepath = FILEPATH + 'hdf/{}.hdf'.format(table)


def get_historical_constituents_wrds_hdf(date_range=None, index='S&P Smallcap 600 Index'):
    # adapted from beat_market_analysis constituent_utils.py
    """
    gets historical constituents from WRDS file

    returns as pandas dataframe with rows as dates and columns as constituent 1, 2, etc
    first gets maximum number of constituents for a given date (so as to make df large enough)
    then converts dicts to df

    common indexes as represented in the idx_ann table:
    SP600: S&P Smallcap 600 Index
    SP400: S&P Midcap 400 Index
    SP500: S&P 500 Comp-Ltd (there's another one with Wed instead of Ltd which I don't know what it is)
    SP1500: S&P 1500 Super Composite

    NASDAQ 100: Nasdaq 100
    """
    idx_names_filename = FILEPATH + 'hdf/names_ix.hdf'
    idx_df = pd.read_hdf(idx_names_filename)
    gvkeyx = idx_df[idx_df['conm'] == index]['gvkeyx'].values
    if len(gvkeyx) > 1:
        print('more than 1 gvkeyx, exiting:')
        print(idx_df[idx_df['conm'] == index])
        return

    gvkeyx = gvkeyx[0]

    idx_hist_filename = FILEPATH + 'hdf/idxcst_his.hdf'
    const_df = pd.read_hdf(idx_hist_filename)

    single_idx_df = const_df[const_df['gvkeyx'] == gvkeyx].copy()
    # combine with securities for ticker symbol
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')
    # abbreviated securities df; only ticker, gvkey, and iid
    sec_short = securities[['tic', 'gvkey', 'iid']]

    single_idx_df = single_idx_df.merge(sec_short, on=['gvkey', 'iid'])

    # create dataframe with list of constituents for each day
    # confusing -- all the changes appear to be at 8pm EST or something like that
    # so shift all days by one
    # first need to convert from and thru columns to plain dates
    # adding offset seems to remove tz, or maybe date does
    single_idx_df['from'] = single_idx_df['from'].apply(lambda x: x.date() + pd.DateOffset(1)).dt.tz_localize('US/Eastern')
    single_idx_df['thru'] = single_idx_df['thru'].apply(lambda x: x.date() + pd.DateOffset(1)).dt.tz_localize('US/Eastern')
    start = single_idx_df['from'].min()
    # get today's date and reset hour, min, sec to 0s
    # latest_m_date = creation_date(idx_hist_filename)
    # today = pd.Timestamp.today(tz='US/Eastern').replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None).tz_localize('US/Eastern')
    # df with last updated dates for tables
    last_updated_fp = FILEPATH + 'last_updated_dates.pk'
    last_updated = pk.load(open(last_updated_fp, 'rb'))
    end = last_updated['idxcst_his']

    # replace NaT with tomorrow's date
    # gives copy warning but can't get rid of it...
    single_idx_df['thru'].fillna(end + pd.DateOffset(days=1), inplace=True)

    nyse = mcal.get_calendar('NYSE')
    # gets all dates
    # date_range = mcal.date_range(start=start, end=end)
    # gets only dates valid for NYSE -- doesn't seem to match historical data
    if date_range is None:
        date_range = nyse.valid_days(start_date=start.date(), end_date=end.date(), tz='US/Eastern')
    else:
        # cutoff at earliest date for index
        date_range = np.array(sorted(date_range))
        date_range = date_range[date_range >= start]


    constituent_companies = OrderedDict()
    constituent_tickers = OrderedDict()
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
        current_tickers = current_stocks['tic']  # company tickers
        constituent_companies[date_string] = current_companies
        constituent_tickers[date_string] = current_tickers
        lengths.append(current_companies.shape[0])


    max_constituents = max(lengths)
    # create dataframe with tickers as values, dates as rows
    col_names = ['c_' + str(i) for i in range(1, max_constituents + 1)]
    all_ticker_data = []
    all_gvkey_data = []
    all_iid_data = []
    for d in tqdm(date_range):
        date_str = d.strftime('%Y-%m-%d')
        # extra columns to fill in blanks where constituents are less than the max
        filler_cols = [None] * (max_constituents - constituent_tickers[date_str].shape[0])
        all_ticker_data.append(constituent_tickers[date_str].values.tolist() + filler_cols)
        all_gvkey_data.append(constituent_companies[date_str]['gvkey'].values.tolist() + filler_cols)
        all_iid_data.append(constituent_companies[date_str]['iid'].values.tolist() + filler_cols)

    ticker_df = pd.DataFrame(index=date_range, data=all_ticker_data, columns=col_names)
    gvkey_df = pd.DataFrame(index=date_range, data=all_gvkey_data, columns=col_names)
    iid_df = pd.DataFrame(index=date_range, data=all_iid_data, columns=col_names)
    # look at number of constituents as a histogram; mostly 600 but a few above and below
    # pd.value_counts(lengths)
    # plt.hist(lengths)
    # plt.show()

    # TODO:
    # need to check that no tickers are used for multiple companies -- will have to combine gvkey and iid with tickers to check

    # get unique dates where changes were made
    unique_dates = set(single_idx_df['from'].unique()) | set(single_idx_df['thru'].unique())

    return constituent_companies, unique_dates, ticker_df, gvkey_df, iid_df


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


def load_small_table(table):
    """
    loads the smaller tables, such as
    funda, sec_shortint, names_ix, security, idxcst_his

    args:
    table -- string; name of table
    df_filepath = FILEPATH + 'hdf/{}.hdf'.format(table)
    """
    df_filepath = FILEPATH + 'hdf/{}.hdf'.format(table)
    return pd.read_hdf(df_filepath)


# TODO: function for merging price data with book value for P/B values of any stock

def portfolio_strategy(index='S&P Smallcap 600 Index', start_date=None):
    """
    tries to implement 20 smallest SPY strategy from paper (see beat_market_analysis github repo)

    from the paper, they have two filters for 'size' and 'P/B'
    assuming the size is a minimum size, and P/B is a maximum

    book value -- use CEQQ (total common equity) from fundq, and cshoq (common shares outstanding) -- ceqq/cshoq
    maybe use rdq as date? or fdateq (final date) -- investigate more
    - rdq is earliest public release date, seems to correspond with SEC filing date
    possibly also use these definitions: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html
    http://financeformulas.net/Book-Value-per-Share.html
    book value per share is BKVLPS from funda table -- to get P/B, take prccd/bkvlps
    try P/B under 1 and 3
    or between 4.5 and 5.0 ?
    https://investinganswers.com/articles/simple-method-calculating-book-value

    steps:
    1. get historical constituents for index
    2. load historical daily price data
    3. get first set of holdings from initialization date
    4. go thru day-by-day getting prices from gvkey/iid; if constituent stops
        existing (e.g. buyout, bankrupt, etc), then sell at last price and buy
        new holding after optional delay
    5.

    . make df with daily close price for each security using gvkey and iid dfs
    .

    specify:
    - index name
    - initialization date
    - rebalance period
    - maybe delay after constituent leaves before entering again
    - option for selling if constituent leaves index due to poor performance
    """
    # merge historical constituents for sp600 with daily price, eps, and market cap data
    # see what returns are on yearly rebalance for 20 smallest marketcap stocks
    # just get first of year dates, then get company market caps
    # get smallest 20 market caps, get close price
    # get close price a year later, calculate overall return
    # repeat ad nauseum

    ### load data
    # index constituents
    constituent_companies, unique_dates, ticker_df, gvkey_df, iid_df = get_historical_constituents_wrds_hdf(index=index)
    unique_dates_dates = [d.date() for d in unique_dates]
    # daily security for prices and market cap
    # current_sec_df = load_secd()  # doesn't contain all securities for some reason
    if index == 'S&P Smallcap 600 Index':
        index_name = 'sp600'

    current_sec_df = pd.read_hdf(FILEPATH + 'hdf/' + index_name + '_constituents_secd.hdf')
    # annual security info for book value
    fundq = load_small_table('fundq')
    # securities listing for delisted reasons (dlrsni)
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')

    # only keep companies that were/are in the index
    constituents = []
    for d in gvkey_df.index:
        constituents.extend(zip(gvkey_df.loc[d].values, iid_df.loc[d].values))

    constituents = set(constituents)
    constituent_list = list(constituents)
    # make column with tuple of gvkey and iid
    current_sec_df['gvkey_iid'] = list(zip(current_sec_df['gvkey'], current_sec_df['iid']))
    fundq['gvkey_iid'] = list(zip(fundq['gvkey'], fundq['iid']))
    securities['gvkey_iid'] = list(zip(securities['gvkey'], securities['iid']))
    # can uncomment this when fix the big secd df
    # sec_df_const = current_sec_df[current_sec_df['gvkey_iid'].isin(constituents)].copy()
    sec_df_const = current_sec_df
    if 'market_cap' not in sec_df_const.columns:
        sec_df_const['market_cap'] = sec_df_const['cshoc'] * sec_df_const['prccd']

    fundq_const = fundq[fundq['gvkey_iid'].isin(constituents)].copy()
    # seems to be a lot of NA values and duplicate rows
    # TODO: check that no quarters are missing for companies after dropping nas
    fundq_const.dropna(subset=['ceqq', 'rdq'], inplace=True)
    # add one day to rdq to be safe
    fundq_const['rdq'] = pd.to_datetime(fundq_const['rdq']).apply(lambda x: x + pd.DateOffset(1)).dt.tz_localize('US/Eastern')
    # EDA
    # fundq_const[['cshoq', 'ceqq', 'rdq', 'fdateq', 'tic', 'gvkey', 'iid', 'cik']
    securities_const = securities[securities['gvkey_iid'].isin(constituents)].copy()
    # explore the delisted reasons
    # securities_const['dlrsni'].unique()
    # securities_const['dlrsni'].value_counts(dropna=False)
    # 01 Acquisition or merger
    # 02 Bankruptcy
    # 03 Liquidation
    # 04 Reverse acquisition (1983 forward)
    # 09 Now a private company
    # 10 Other (no longer files with SEC among other reasons)
    # 20 Other (issue-level activity; company remains active on the file)

    ### EDA on bankruptcy
    # look at a security that went bankrupt to see what price did
    # securities_const[securities_const['dlrsni'] == '02']
    # check out BLGM (024415, 01) stock
    # joined index 1994-10-01, left 2008-08-19
    # huge spike and decrease around 2005-08
    # hopefully would've rebalanced out in mid 2000s
    # import matplotlib.pyplot as plt
    # sec_df_const.set_index('datadate', inplace=True)
    # sec_df_const[sec_df_const['gvkey_iid'] == ('024415', '01')]['prccd'].plot(); plt.show()

    # doesn't quite work...so close
    """
    fundq_const.sort_values(by=['gvkey_iid', 'rdq'], inplace=True)
    sec_df_const.sort_values(by=['gvkey_iid', 'datadate'], inplace=True)

    sec_df_const_pb = pd.merge_asof(sec_df_const,
                                    fundq_const[['ceqq', 'rdq', 'gvkey_iid']],
                                    left_on='datadate',
                                    right_on='rdq',
                                    by='gvkey_iid',
                                    direction='backward')
    """

    ### adds quarterly stockholders' equity to daily price data
    # with sp600, took about 32 mins (about 2150 constituents)
    all_merged = []
    not_in_fundq = []
    nat = np.datetime64('NaT')
    for gvkey_iid in tqdm(sec_df_const['gvkey_iid'].unique().tolist()):
        fundq_sm = fundq_const[fundq_const['gvkey_iid'] == gvkey_iid][['rdq', 'ceqq']].sort_values(by='rdq')
        sec_df_const_sm = sec_df_const[sec_df_const['gvkey_iid'] == gvkey_iid].sort_values(by='datadate')
        if fundq_sm.shape[0] == 0:
            print(gvkey_iid)
            not_in_fundq.append(gvkey_iid)
            # fill in np.nan and np.nat for rdq and ceqq
            merged = sec_df_const_sm.copy()
            merged['rdq'] = nat
            merged['rqd'] = merged['rdq'].dt.tz_localize('US/Eastern')
            merged['ceqq'] = np.nan
        else:
            merged = pd.merge_asof(sec_df_const_sm,
                                    fundq_sm,
                                    left_on='datadate',
                                    right_on='rdq',
                                    direction='backward')

        all_merged.append(merged)

    full_df = pd.concat(all_merged)
    full_df['pb_ratio'] = full_df['market_cap'] / full_df['ceqq']
    # replace with what was largest value for sp600
    full_df['pb_ratio'] = full_df['pb_ratio'].replace(np.inf, 1.111e+11)
    full_df['pb_ratio'].hist(log=True, bins=50); plt.show()
    full_df[(full_df['pb_ratio'] < 10) & (full_df['pb_ratio'] > -10)]['pb_ratio'].hist(bins=50); plt.show()

    # takes a while...
    full_df['datadate'] = full_df['datadate'].dt.date

    # start at earliest date by default
    if start_date is None:
        start_date = min(unique_dates).date()

    # trimmed_df = full_df[full_df['datadate'] >= start_date]
    # earliest market cap seems to be 1998-4-1
    # just start at 2k like the paper
    trimmed_df = full_df[full_df['datadate'] >= pd.to_datetime('2000-01-01').date()].copy()
    # for testing
    trimmed_df = trimmed_df[trimmed_df['datadate'] <= pd.to_datetime('2003-01-01').date()].copy()



    def filter_securities(book_val_max=3, min_size=None, n_smallest=20):
        """
        Filters securities by a set of criterion.
        This first gets most recent book value, then filters by book val
        under the book_val_max.
        Then filters by size: minimum size of min_size (based on market cap).
        Lastly, gets the n_smallest smallest companies by market cap.
        """
        pass

    # need a constituent companies dict that can match the tuple format
    const_comp_tuples = {}
    for d in constituent_companies.keys():
        const_comp_tuples[d] = list(zip(constituent_companies[d]['gvkey'], constituent_companies[d]['iid']))


    ## first try with just 20 smallest (greater than 17M mkt cap)
    # get initial portfolio holdings
    n_smallest = 20
    holdings = []
    holding_prices = []
    first = True


    for d in tqdm(sorted(trimmed_df['datadate'].unique())):
        # set first holdings first time around
        if first:
            # get prices from that date
            current = trimmed_df[trimmed_df['datadate'] == d].copy()
            # only keep securities currently in the index
            current = current[current['gvkey_iid'].isin(set(const_comp_tuples[d.strftime('%Y-%m-%d')]))].copy()
            current_filt = current[current['market_cap'] >= 17e6].copy()
            # take n smallest
            current_filt.sort_values(by='market_cap', inplace=True)
            # holding_prices.append(current_filt.iloc[:n_smallest]['prccd'].values.tolist())
            hold = current_filt.iloc[:n_smallest]['gvkey_iid'].values.tolist()
            holdings.append(hold)
            current_year = r['datadate'].year
            current_holdings = hold
            hold_price_temp = []
            for c in current_holdings:
                one_security = current_filt[current_filt['gvkey_iid'] == c]
                hold_price_temp.append(one_security['prccd'].values[0])

            holding_prices.append(hold_price_temp)

            first = False
            continue

        if r['datadate'].year == current_year:
            # append prices; TODO: if security no longer in index replace with new one
            current = trimmed_df[trimmed_df['datadate'] == d]
            hold_price_temp = []
            for c in current_holdings:
                if c in const_comp_tuples[d.strftime('%Y-%m-%d')]:
                    one_security = current_filt[current_filt['gvkey_iid'] == c]
                    hold_price_temp.append(one_security['prccd'].values[0])
                else:
                    hold_price_temp.append(np.nan)

            holding_prices.append(hold_price_temp)
            continue
        else:
            # rebalance
            current = trimmed_df[trimmed_df['datadate'] == d]
            current_filt = current[current['market_cap'] >= 17e6].copy()
            # take n smallest
            current_filt.sort_values(by='market_cap', inplace=True)
            # holding_prices.append(current_filt.iloc[:n_smallest]['prccd'].values.tolist())
            hold = current_filt.iloc[:n_smallest]['gvkey_iid'].values.tolist()
            holdings.append(hold)
            current_year = r['datadate'].year
            current_holdings = hold
            hold_price_temp = []
            for c in current_holdings:
                one_security = current_filt[current_filt['gvkey_iid'] == c]
                hold_price_temp.append(one_security['prccd'].values[0])

            holding_prices.append(hold_price_temp)


    holding_prices_df = pd.DataFrame(index=sorted(trimmed_df['datadate'].unique()), data=holding_prices)
    # evaluate returns
