#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import sys
import os

# if  not os.path.abspath('./') in sys.path:
#     sys.path.append(os.path.abspath('./'))
# if  not os.path.abspath('../') in sys.path:
#     sys.path.append(os.path.abspath('../'))


import datetime
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as pdr
import yfinance as yf
import traceback
import json
import requests
import financialmodelingprep as fprep

import pyarrow as pa
import redis

import time
import tqdm
import argparse

import schedule_it#@UnresolvedImport


fmp_key = open('temp_folder/fmp_key.txt','r').read()

def dt_to_yyyymmdd(d):
    return int(d.year)*100*100 + int(d.month)*100 + int(d.day)

def str_to_yyyymmdd(d,sep='-'):
    try:
        dt = datetime.datetime.strptime(str(d)[:10],f'%Y{sep}%m{sep}%d')
    except:
        return None
    s = '%04d%02d%02d' %(dt.year,dt.month,dt.day)
    return int(s)

def str_to_date(d,sep='-'):
    try:
        dt = datetime.datetime.strptime(str(d)[:10],f'%Y{sep}%m{sep}%d')
    except:
        return None
    return dt


def fetch_history(symbol,dt_beg,dt_end):
    df = yf.download(symbol, dt_beg, dt_end,threads=False)
    # move index to date column, sort and recreate index
    df['date'] = df.index
    df = df.sort_values('date')
    df.index = list(range(len(df)))
    # make adj close the close
    df = df.drop(['Adj Close'],axis=1)
    cols = df.columns.values 
    cols_dict = {c:c[0].lower() + c[1:] for c in cols}
    df = df.rename(columns = cols_dict)
    df['settle_date'] = df.date.apply(str_to_yyyymmdd)
    df = df.groupby('settle_date',as_index=False).first()
    return df

# def get_port_info_values(syms):
#     names = syms if type(syms)==list else syms.tolist()

#     tickers = yf.Tickers(names)
#     dict_list = []
#     for n in names:
#         try:
#             d = tickers.tickers[n].get_info()
#             d['symbol'] = n
#             dict_list.append(d)
#         except Exception as e:
#             print(f'Exception on symbol {n}: {str(e)[0:100]}')
#     df_info_values = pd.DataFrame(dict_list)
#     return df_info_values
    
def get_port_info_values(syms):
    names = syms if type(syms)==list else syms.tolist()
    dfp = fprep.fmp_profile(syms)
    dfr = fprep.fmp_ratios(syms)
    df_info_values = dfp.merge(dfr,on='symbol',how='left')
    df_info_values['pe'] = df_info_values.peRatioTTM
    return df_info_values

def update_wf_port_info(syms,fundamental_key):
    try:
        df_info_values = get_port_info_values(syms)
        if len(df_info_values)>0:
            info_values_key = fundamental_key
            update_redis_df(info_values_key,df_info_values)
    except Exception as e:
        traceback.print_exc()


def update_redis_df(key,df):
    context = pa.default_serialization_context()#@UndefinedVariable
    redis_db.set(key, context.serialize(df).to_buffer().to_pybytes())


def get_fmp_ratios(symbol):
    ratios_url = f'https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={fmp_key}'
    response = json.loads(requests.get(ratios_url).text)
    return response

def update_db(beg_sym=None,port_path=None,fundamental_key=None):
    syms = None
    if port_path is not None:
        syms = pd.read_csv(port_path).symbol.values
    else:
        sp_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df_sp_members = pd.read_csv(sp_url)  
        df_sp_members = df_sp_members.sort_values('Symbol')
        if beg_sym is not None:
            df_sp_members = df_sp_members[df_sp_members.Symbol>=beg_sym]
        syms = df_sp_members.Symbol.values
    syms = np.append(syms,['SPY','QQQ'])
    data_end_date = datetime.datetime.now()
    data_beg_date = data_end_date - relativedelta(years=5)
    pe_values = []
    closes = []
    with tqdm.tqdm(syms,position=0,leave=True) as pbar:
        for sym in pbar:
            pbar.set_postfix_str(s=sym)
            try:
                df_temp = fetch_history(sym, data_beg_date, data_end_date)
                update_redis_df(f'{sym}_csv',df_temp)
            except Exception as e:
                print(f"ERROR on {sym}: {str(e)}")
        
    if fundamental_key is not None:
        print(f"Fetching fundamental values for {len(syms)} securites")
        update_wf_port_info(syms,fundamental_key)


def schedule_updates(t=8,unit='hour',beg_sym=None,port_path=None,
    num_runs=None,fundamental_key=None):
    logger = schedule_it.init_root_logger("logfile.log", "INFO")
    counter = num_runs
    while True:
        logger.info(f"scheduling update for {unit} {t}")
        sch = schedule_it.ScheduleNext(unit, t,logger = logger)
        sch.wait()
        logger.info(f"updating history")
        update_db(beg_sym=beg_sym,port_path=port_path,fundamental_key=fundamental_key)
        if counter is not None:
            counter = counter - 1
            if counter <=0:
                return
        logger.info(f"sleeping until next {t} {unit} before next scheduling")
        time.sleep(5*60)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog = 'redis_server_stock_data_updates',
        description = 'Update the redis server with new DataFrames for S&P500 stocks',
        )
    hour = datetime.datetime.now().hour
    parser.add_argument('--port',default=6379,type=int,help="redis port") 
    parser.add_argument('--portfolio_path',default=None,type=str,
        help="portfolio path to obtain symbols.  The symbols should only be 'underlying' symbols") 
    parser.add_argument('--timevalue',default=hour,type=int,
        help="an int value that is either hours or minutes, depending on the arg 'unit'")
    parser.add_argument('--unit',default='hour',type=str,help="either 'hour' or 'minute'")  
    parser.add_argument('--numruns',default=100,type=int,help="number of times to loop")
    fundamental_help = """
    If fundamental_key is provided, then, that redis key will be updated with the 
    fundamental information for the symbols in the portfolio
    """

    parser.add_argument('--fundamental_key',default=None,type=str,
        help=fundamental_help)
    args = parser.parse_args()  
    redis_port = args.port
    redis_db = redis.Redis(host = 'localhost',port=redis_port,db=0)

    # t = 20 if len(sys.argv)<2 else int(sys.argv[1])
    # bs = None if len(sys.argv)<3 else sys.argv[2]
    # port_path = None if len(sys.argv)<4 else sys.argv[3]
    # unit = 'hour' if len(sys.argv)<5 else sys.argv[4]
    # num_runs = 100 if len(sys.argv)<6 else int(sys.argv[5])
    t = args.timevalue
    bs = None
    port_path = args.portfolio_path
    unit = args.unit
    num_runs =args.numruns
    fundamental_key = args.fundamental_key
    print(args)
    # fundamental_key is often 'wf_port_info_csv'
    schedule_updates(t=t,unit=unit,beg_sym=bs,port_path=port_path,
        num_runs=num_runs,fundamental_key=fundamental_key)



