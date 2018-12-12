import pandas as pd

FILEPATH = '/home/nate/Dropbox/data/wrds/compustat_north_america/'


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
    idx_df = pd.read_hdf(FILEPATH + 'hdf/names_ix.hdf')
    gvkeyx = idx_df[idx_df['conm'] == index]['gvkeyx'].values
    if len(gvkeyx) > 1:
        print('more than 1 gvkeyx, exiting:')
        print(idx_df[idx_df['conm'] == index])
        return

    gvkeyx = gvkeyx[0]

    const_df = pd.read_hdf(FILEPATH + 'hdf/idxcst_his.hdf')

    single_idx_df = const_df[const_df['gvkeyx'] == gvkeyx].copy()
    # combine with securities for ticker symbol
    securities = pd.read_hdf(FILEPATH + 'hdf/security.hdf')
    # abbreviated securities df; only ticker, gvkey, and iid
    sec_short = securities[['tic', 'gvkey', 'iid']]

    single_idx_df = single_idx_df.merge(sec_short, on=['gvkey', 'iid'])

    # create dataframe with list of constituents for each day
    start = single_idx_df['from'].min()
    # get today's date and reset hour, min, sec to 0s
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
    col_names = ['c_' + str(i) for i in range(max_constituents)]
    df = pd.DataFrame(columns=col_names)
    for d in tqdm(constituent_tickers.keys()):
        ticker_data = constituent_tickers[d].values.tolist() + [None] * (max_constituents - constituent_tickers[d].shape[0])
        df.append(pd.DataFrame(index=[d], data=[ticker_data], columns=col_names))
    # look at number of constituents as a histogram; mostly 600 but a few above and below
    # pd.value_counts(lengths)
    # plt.hist(lengths)
    # plt.show()

    # TODO:
    # need to check that no tickers are used for multiple companies

    # get unique dates where changes were made
    unique_dates = set(single_idx_df['from'].unique()) | set(single_idx_df['thru'].unique())

    return constituent_companies, unique_dates


most_constituents =
df = pandas.DataFrame(constituent_companies)
