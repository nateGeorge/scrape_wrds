import os

import wrds

uname = os.environ.get('wrds_username')
password = os.environ.get('wrds_password')

db = wrds.Connection(wrds_username=unam)
# saves credentials
db.create_pgpass_file()

db.list_libraries()
db.list_tables('zacks')
db.list_tables('ciq')  # don't have permission??
db.list_tables('comp_global_daily')

db.list_tables('comph')

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
