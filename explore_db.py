import os
import gc

import pandas as pd
import wrds

FILEPATH = '/home/nate/Dropbox/data/wrds/compustat_north_america/'

hdf_settings = {'key': 'data',
                'mode': 'w',
                'complib': 'blosc',
                'complevel': 9}

db = wrds.Connection()
# saves credentials
# db.create_pgpass_file()

db.list_libraries()
db.list_tables('zacks')
db.list_tables('ciq')  # don't have permission??
db.list_tables('comp_global_daily')

db.list_tables('comp')


def download_entire_table(tablename, library='comp'):
    """
    default library is the compstat lib
    download entire table
    e.g. tablename='sec_shortint'

    tables downloaded 9-12:
    sec_shortint
    security
    secd


    secd is about 39GB in a pandas df...

    TODO: get latest date already downloaded and use sql query to update

    for tables like 'security', check if any more rows and grab new stuff, or just grab whole table if cant figure out what new stuff is

    """
    nrows = db.get_row_count(library, tablename)
    print('number of rows:', nrows)
    #db.describe_table(library, tablename)
    # nrows = 1000000
    if tablename == 'secd':
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

        nobs = 10000000
        for i, start in enumerate(range(0, nrows, nobs), 1):
            print('on part', str(i))
            df = db.get_table(library, tablename, columns=cols_to_use, obs=nobs, offset=start)
            df.to_hdf(FILEPATH + 'hdf/{}.hdf'.format(tablename + '_min_part_' + str(i)), **hdf_settings)
            del df
            gc.collect()
    else:
        df = db.get_table(library, tablename, obs=nrows)
        df.to_hdf(FILEPATH + 'hdf/{}.hdf'.format(tablename), **hdf_settings)


def load_and_combine_sec_dprc():
    dfs = []
    for i in range(1, 13):
        print(i)
        dfs.append(pd.read_hdf(FILEPATH + 'hdf/sec_dprc_min_part_{}.hdf'.format(str(i))))

    df = pd.concat(dfs)


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
