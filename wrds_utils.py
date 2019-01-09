import os
import platform
import datetime
from pytz import timezone
from collections import OrderedDict

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
    start = single_idx_df['from'].min()
    # get today's date and reset hour, min, sec to 0s
    # TODO: think about other ways to handle end date
    latest_m_date = creation_date(idx_hist_filename)
    today = pd.Timestamp.today(tz='US/Eastern').replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None).tz_localize('US/Eastern')
    end = eastern.localize(latest_m_date)

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


def portfolio_strategy(index='S&P Smallcap 600 Index'):
    """
    tries to implement 20 smallest SPY strategy from paper (see beat_market_analysis github repo)

    from the paper, they have two filters for 'size' and 'P/B'
    assuming the size is a minimum size, and P/B is a maximum

    book value per share is BKVLPS from funda table -- to get P/B, take prccd/bkvlps
    try P/B under 1 and 3

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

    # index constituents
    constituent_companies, unique_dates, ticker_df, gvkey_df, iid_df = get_historical_constituents_wrds_hdf(index=index)
    # daily security for prices and market cap
    current_sec_df = load_secd()
    # annual security info for book value
    funda = load_small_table('funda')
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
