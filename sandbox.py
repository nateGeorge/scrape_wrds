
hdf_settings = {'key': 'data',
                'mode': 'w',
                'complib': 'blosc',
                'complevel': 9}

fund_hdf_settings = {'key': 'data',
                    'mode': 'w',
                    'complevel': 9}



short_df = pd.read_hdf(FILEPATH + 'hdf/short_interest_9-12-2018.hdf')
short_df[['Ticker Symbol', 'Short Interest Settlement Date', 'Shares Held Short as of SettlementDate']]
# TODO: rename columns with shorter names, remove NANs especially in tickers, and figure out adjustments
# Also match up industry code with industry from finviz/yahoo etc

# when trying to dl the whole daily dataset, problems...
# 2 years should be about 5 GB
# problem: lots of repeated data, such as website, address, description, etc, which is taking up a bunch of memory.
# need to get just the address/etc from the table for each company, and not
daily_df = pd.read_hdf(FILEPATH + 'hdf/daily_security_data_9-12-2018.hdf')
daily_df['Market Cap'] = daily_df['Shares Outstanding'] * daily_df['Price - Close - Daily']
'TLRY' in daily_df['Ticker Symbol']

daily_df_17 = pd.read_csv(FILEPATH + 'tsv/daily_securities_2017-9-12-2018.txt', sep='\t')

"""
# takes a very long time to read this -- goddam excel
# daily security data
df = pd.read_excel(FILEPATH + 'daily_security_data_9-12-2018.xlsx')
df.to_hdf(FILEPATH + 'daily_security_data_9-12-2018.hdf', **hdf_settings)

# short data
short_df = pd.read_excel(FILEPATH + 'short_interest_9-12-2018.xlsx')
short_df.to_hdf(FILEPATH + 'hdf/short_interest_9-12-2018.hdf', **hdf_settings)

# quarterly industry data
ind_quarterly_df = pd.read_csv(FILEPATH + 'tsv/industry_quarterly_9-12-2018.txt', sep='\t')
ind_quarterly_df.to_hdf(FILEPATH + 'hdf/industry_quarterly_9-12-2018.hdf', **hdf_settings)

# industry annual
ind_ann_df = pd.read_csv(FILEPATH + 'tsv/industry_annual_9-12-2018.txt', sep='\t')
ind_ann_df.to_hdf(FILEPATH + 'hdf/industry_annual_9-12-2018.hdf', **hdf_settings)
"""

# # quarterly fundamental data
# qrt_fund_df = pd.read_csv(FILEPATH + 'tsv/quarterly_fundamentals_9-12-2018.txt', sep='\t')
# # blosc doesn't seem to work
# qrt_fund_df.to_hdf(FILEPATH + 'hdf/quarterly_fundamentals_9-12-2018.hdf', **fund_hdf_settings)
#
# # annual fundamentals
# ann_fund_df = pd.read_csv(FILEPATH + 'tsv/annual_fundamentals_9-12-2018.txt', sep='\t')
# ann_fund_df.to_hdf(FILEPATH + 'hdf/annual_fundamentals_9-12-2018.hdf', **fund_hdf_settings)
#

"""
# monthly securities data
mnth_sec_df = pd.read_csv(FILEPATH + 'tsv/monthly_securities_data_9-12-2018.txt', sep='\t')
mnth_sec_df.to_hdf(FILEPATH + 'hdf/monthly_securities_data_9-12-2018.hdf', **hdf_settings)

# otc daily
otc_daily_df = pd.read_excel(FILEPATH + 'excel/otc_EOD_9-12-2018.xlsx')
otc_daily_df.to_hdf(FILEPATH + 'hdf/otc_EOD_9-12-2018.hdf', **hdf_settings)

# index daily prices
idx_pr_daily_df = pd.read_csv(FILEPATH + 'index_prices_daily_9-12-2018.txt', sep='\t')
idx_pr_daily_df.to_hdf(FILEPATH + 'hdf/index_prices_daily_9-12-2018.hdf', **hdf_settings)

# index constituents
idx_const_df = pd.read_csv(FILEPATH + 'index_constituents_9-12-2018.txt', sep='\t')
idx_const_df.to_hdf(FILEPATH + 'hdf/index_constituents_9-12-2018.hdf', **hdf_settings)
"""
