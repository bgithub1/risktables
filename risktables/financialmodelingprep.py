#!/usr/bin/env python
# coding: utf-8

# ## Use financialmodelingprep to obtain fundemental stock info
# See the api docs at: https://financialmodelingprep.com/developer/docs/
# 
# The financialmodelingprep.com api allows you to capture useful stock fundemental information with 
# 

# In[63]:


import pandas as pd
import numpy as np
import requests
import json
from IPython import display
import pdb
import argparse



fmp_key = open('temp_folder/fmp_key.txt','r').read()

def get_fmp_json(symbol_list,route):
    fmp_json_array = []
    for i in np.arange(0,len(symbol_list),3):
        clist_string = ','.join([c.upper() for c in symbol_list[i:i+3]])
        fmp_url = f'https://financialmodelingprep.com/api/v3/{route}/{clist_string}?apikey={fmp_key}'
        frs =  requests.get(fmp_url).json()
        if (type(frs)==list):
            fmp_json_array.extend(frs)
        else:
            fmp_json_array.append(frs)
    return fmp_json_array
    
def fmp_profile(symbol_list):
    '''
    Get profile data for list
    :param symbol_list: a list like ['aapl','msft','amzn','meta','googl']
    
    
    '''
    route = 'profile'
    df_final = pd.DataFrame()
    fmp_json_array = get_fmp_json(symbol_list,route)
    for fmp_json in fmp_json_array:
        dft = pd.json_normalize(fmp_json)
        df_final = pd.concat([df_final,dft])
    return df_final 

def fmp_ratios(symbol_list):
    '''
    Get ratio data for list
    :param symbol_list: a list like ['aapl','msft','amzn','meta','googl']
    '''
    route = 'financial-ratios'
    df_final = pd.DataFrame()
    fmp_json_list = get_fmp_json(symbol_list,route)
    for fmp_json in fmp_json_list:
        if 'ratiosList' not in fmp_json:
            continue
        json2 = fmp_json['ratiosList']
        dft = pd.json_normalize(json2,meta=['symbol'],record_path=['ratios'])
        df_final = pd.concat([df_final,dft])
    return df_final 


# In[60]:


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'financialmodelingprep',
        description = 'Get stock profiles and ratios using financialmodelingprep API',
        )
    parser.add_argument('--portfolio_path',default=None,type=str,help="portfolio path to obtain symbols") 
    args = parser.parse_args() 
    port_path = args.portfolio_path
    if port_path is not None:
        stocks = pd.read_csv(port_path).symbol.values
    else:
        stocks = pd.read_csv('spdr_etfs.csv').symbol.values
    dfp = fmp_profile(stocks)
    dfr = fmp_ratios(stocks)
    print('profile')
    display.display(dfp)
    print('ratios')
    display.display(dfr)    


