#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import sys


# In[2]:


def get_yahoo_data(sym,beg_date,end_date):
    try:
        df = yf.download(sym, beg_date, end_date)
        return df
    except:
        return None


# In[3]:


def create_spdr_options(edate=datetime.datetime.now()):
    df_spdr_stocks = pd.read_csv('spdr_stocks.csv')
    names = [n[0:3] for n in df_spdr_stocks.symbol.values]
    edate = datetime.datetime.now()
    bdate = edate - datetime.timedelta(20)
    y = (edate + datetime.timedelta(60)).year
    base_amt = 10000
    closes = []
    pc_list = list(np.array([['c','p'] for _ in range(5)]).reshape(-1))
    for i,n in enumerate(names):
        df = get_yahoo_data(n,bdate,edate)
        last_close = df.iloc[-1].Close
        strike = int(last_close.round(0))
        yyyymmdd = y*100*100+1231
        pc = pc_list[i]
        position = int(round(base_amt/strike,0))
        options_symbol = f"{n}_{yyyymmdd}_{strike}_{pc}"
        closes.append(
            {
                'symbol':options_symbol,
                'position':position
            }
        )
    df_spdr = pd.DataFrame(closes)
#     df_spdr['symbol'] = df_spdr['symbol'] + '_' + df_spdr.yyyymmdd.astype(str) + '_' + df_spdr.pc
#     df_spdr = df_spdr[['symbol','position']].copy()
    return df_spdr        



# In[4]:


if __name__== '__main__':
    df_spdr_options = create_spdr_options()
    print(df_spdr_options)
    if '.csv' in sys.argv[1] is not None:
        csv_file_path = sys.argv[1]
        print(f'writing file to {csv_file_path}')
        df_spdr_options.to_csv(csv_file_path, index=False)
        


# In[ ]:





# In[ ]:




