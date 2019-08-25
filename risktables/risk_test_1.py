'''
Created on Jul 23, 2019

@author: bperlman1
'''
import sys,os
if  not './' in sys.path:
    sys.path.append(os.path.abspath('./'))
if  not '../' in sys.path:
    sys.path.append(os.path.abspath('../'))

import argparse as ap
from risktables import risk_tables
import pandas as pd

DEFAULT_PORTFOLIO_NAME = 'spdr_stocks.csv'


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--initial_portfolio',type=str,default=DEFAULT_PORTFOLIO_NAME,help='initial portfolio to Load')
    parser.add_argument('--use_postgres',
        default=False,action='store_true',
        help='set to True if using Postgres db for history data')
    parser.add_argument('--postgres_info_path',type=str,
        help='path to postgres_info.csv file that holds postgres login, schema and table info for each db',
        default='./postgres_info.csv')
    parser.add_argument('--calculate_hedge_ratio',
        default=False,action='store_true',
         help='set to True if calculating hedge portfolio using Sector Spdr ETFs')
                
    args = parser.parse_args()
    
    # Step 1: get postgres info
    df_pginfo = pd.read_csv(args.postgres_info_path).fillna('')
    #   get info for dashrisk
    s = df_pginfo.loc[df_pginfo.config_name=='dashrisk_local']
    
    # Step 2: create an instance of RiskCalcs
    rt = risk_tables.RiskCalcs(
        use_postgres=args.use_postgres, 
        dburl=s.dburl.values[0], 
        databasename=s.databasename.values[0], 
        username=s.username.values[0], 
        password=s.password.values[0], 
        schema_name=s.schema_name.values[0], 
        yahoo_daily_table=s.table_names.values[0],
        calculate_hedge_ratio=args.calculate_hedge_ratio)
    
    # Step 3: get an actual portfolio
    df_portfolio = pd.read_csv(args.initial_portfolio)
    risk_dict_dfs = rt.calculate(df_portfolio)
    for k in risk_dict_dfs:
        print(k)
        dict_item = risk_dict_dfs[k]
        object_to_print = dict_item
        if type(dict_item)==dict:
            if k[:2] == 'df':
                object_to_print = risk_tables.make_df(risk_dict_dfs[k])
        print(object_to_print)

       