'''
Created on Mar 1, 2019

@author: bperlman1
'''
import sys,os
if  not './' in sys.path:
    sys.path.append('./')
if  not '../' in sys.path:
    sys.path.append('../')

import pandas as pd
from risktables import portfolio_hedge as phedge
import argparse as ap




if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--use_spy',type=bool,
                        help='Use SPY as your portfolio, otherwise use 20 randomly created members of SP 500, with random weights. (Default = False)',
                        default=False)
    parser.add_argument('--refetch_data',type=bool,
                        help='Re-fetch all data. (Default = False)',
                        default=False)
    parser.add_argument('--max_hedge_symbols',type=int,
                        help='Maximum number of symbols from sector spdrs that you will use to hedge. (Default = 5)',
                        default = 5)
    parser.add_argument('--rounding_value_for_hedge_comarisons',type=float,
                        help='Round all absolute differences to this percent, so that simpler portfolios can end up being the best. (Default = .002)',
                        default = .001)
    args = parser.parse_args()

#     use_min_variance = args.use_min_variance
    use_spy = args.use_spy
    refetch_data = args.refetch_data
    max_hedge_symbols = args.max_hedge_symbols
    if max_hedge_symbols > 10:
        max_hedge_symbols = 10
    rounding_value_for_hedge_comarisons = args.rounding_value_for_hedge_comarisons
    
    if use_spy:
        portfolio_column_name = 'SPY'
        df = phedge.fetch_sector_spdr_df(refresh=refetch_data)
    else:
        portfolio_column_name = 'port'
        if refetch_data:
            df = phedge.create_random_portfolio_history()
        else:
            df = pd.read_csv(phedge.RANDOM_PORTFOLIO_PATH)

    best_ph = phedge.best_hedge(df,portfolio_column_name,max_hedge_symbols,rounding_value_for_hedge_comarisons)    
    best_ph.plot_hedge_ratios_vs_real()
    print(best_ph.hedge_ratio_dict)
