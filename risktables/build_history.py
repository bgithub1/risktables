'''
Created on Feb 16, 2019

Use the main in this module to build an history sql database, by instantiating an
  instance of HistoryBuilder.

Usage: (make sure your virtualenv has all the dependencies in ../requirements.txt)

1. Build database from scratch, using symbols from the SP 500, the sector spdr ETF's 
   and the commodity ETFs
$ python3 build_history.py --delete_schema True --fetch_from_yahoo True --build_table True

2. Update the existing symbols in the database
$ python3 build_history.py --update_table  True

3. Delete the existing table, and recreate it
$ python3 build_history.py --delete_table --True --fetch_from_yahoo True --build_table True


@author: bperlman1
'''
import argparse as ap
import sys
import os
if  not './' in sys.path:
    sys.path.append('./')
if  not '../' in sys.path:
    sys.path.append('../')
from risktables import pg_pandas as pg
from os import listdir
from os.path import isfile, join
# import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import datetime as dt
import time
from risktables import logger_init as li
from risktables import barchart_api as bcapi
import numpy as np

DEFAULT_DAYS_TO_FETCH = 120

def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

def get_last_business_day(date):
    d = date
    for _ in range(5):
        if is_business_day(d):
            return d
        d = d - dt.timedelta(1)
    return None

class HistoryBuilder():
    def __init__(self,
                 delete_schema=False,
                 delete_table=False,
                 fetch_from_yahoo=False,
                 build_table=False,
                 update_table=True,
                 beg_date=None,
                 end_date=None,
                 dburl=None,
                 databasename=None,
                 username='',
                 password='',
                 schema_name=None,
                 yahoo_daily_table=None,
                 initial_symbol_list=None,
                 days_to_fetch=DEFAULT_DAYS_TO_FETCH,
                 temp_folder='./temp_folder',
                 logger=None,
                 STOCKS_DIR = None,
                 use_datahub=False):
        self.delete_schema = delete_schema
        self.delete_table = delete_table
        self.fetch_from_yahoo = fetch_from_yahoo
        self.build_table = build_table
        self.update_table = update_table
        self.beg_date = beg_date
        self.end_date = end_date
        self.temp_folder = temp_folder
        self.STOCKS_DIR = f'{temp_folder}/stocks' if STOCKS_DIR is None else STOCKS_DIR
        self.bch = self.get_barchart_api()
        self.logger = logger if logger is not None else li.init_root_logger('logfile.log', 'INFO')
        self.dburl = dburl if dburl is not None else 'localhost'
        self.username = '' if username is None else username 
        self.password = '' if password is None else password
        self.databasename = databasename if databasename is not None else 'testdb'
        self.schema_name = schema_name if schema_name is not None else 'test_schema'
        self.yahoo_daily_table = yahoo_daily_table if yahoo_daily_table is not None else 'yahoo_daily'
        self.pga = pg.PgPandas(databasename=self.databasename,username=self.username,password=self.password,dburl=self.dburl)
        self.full_table_name = self.schema_name + '.' + self.yahoo_daily_table
        self.initial_symbol_list = self.get_sp_stocks() if initial_symbol_list is None else initial_symbol_list
        self.days_to_fetch = days_to_fetch
        self.use_datahub = use_datahub
        
    def write_hist_dict_to_csv(self,hist_dict):
        try:
            os.makedirs(self.STOCKS_DIR)
        except:
            pass
        for sym,df in hist_dict.items():
            if df is None or len(df)<=0:
                continue
            csv_path = f'{self.STOCKS_DIR}/{sym}.csv'
            self.logger.info(f'writing {sym} to {csv_path}')
            df['Date'] = df.index
            df.index = range(len(df))
            df.to_csv(csv_path,index=False)
            
            
    def build_history_dict(self):
        symbols = self.initial_symbol_list
        hist_dict = {}
        end_date = dt.datetime.now()
        beg_date = end_date - dt.timedelta(self.days_to_fetch)
        for sym in symbols:
            df = None
            self.logger.info(f'processing {sym}')
            df = self.get_yahoo_data(sym, beg_date, end_date)
#             try:
#                 df =web.DataReader(sym, 'yahoo', beg_date, end_date)
#             except:
#                 try:
#                     df =web.DataReader(sym, 'yahoo', beg_date, end_date)
#                 except Exception as e:
#                     self.logger.warn(str(e))
#                     continue
            hist_dict[sym] = df
            time.sleep(.5)
        return hist_dict
    
    def get_yahoo_data(self,sym,beg_date,end_date):
        try:
#             df =web.DataReader(sym, 'yahoo', beg_date, end_date)
            df = yf.download(sym, beg_date, end_date)
            return df
        except:
            try:
#                 df =web.DataReader(sym, 'yahoo', beg_date, end_date)
                df = yf.download(sym, beg_date, end_date)
                return df
            except:
                try:
                    df = self.get_barchart_data(sym, beg_date, end_date)
                    return df
                except Exception as e:
                    self.logger.warn(str(e))
        return None

    def get_barchart_api(self):
        # set this to 'free' or 'paid'
        endpoint = 'free' # free or paid
        
        # set the bar_type and the interval
        bar_type='daily' # minutes, daily, monthly
        interval=1 # 1,5,15,30,60
        
        # create an instance 
#         api_key = open(f'./temp_folder/{endpoint}_api_key.txt','r').read()
        api_key = open(f'./{self.temp_folder}/{endpoint}_api_key.txt','r').read()
        endpoint_type=f'{endpoint}_url'
        bch = bcapi.BcHist(api_key, bar_type=bar_type, interval=interval,endpoint_type = endpoint_type)
        return bch
            
    def get_barchart_data(self,sym,beg_date,end_date):
        try:
            beg_yyyymmdd = '%04d%02d%02d' %(beg_date.year,beg_date.month,beg_date.day)
            end_yyyymmdd = '%04d%02d%02d' %(end_date.year,end_date.month,end_date.day)
            tup = self.bch.get_history(sym, beg_yyyymmdd, end_yyyymmdd)
            df = tup[1]
            df2 = self.convert_barchart_to_yahoo_format(df)
            return df2
        except Exception as e:
            self.logger.warn(str(e))
        return None

    def convert_barchart_to_yahoo_format(self,df):
        df2 = df.copy()
        df2.index = df2.tradingDay.apply(lambda d: pd.Timestamp(d))
        df2.index.name = 'Date'
        newcols = {c:c[0].upper()+c[1:] for c in df2.columns.values}
        df3 = df2.rename(columns=newcols)
        df3 = df3[['High','Low','Open','Close','Volume']]
        df3['Adj Close'] = df3.Close
        return df3
    
    def get_sp_stocks(self):
        url_constituents = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        spydr_short_names = ['SPY','XLE','XLU','XLK','XLB','XLP','XLY','XLI','XLC','XLV','XLF']
        commodity_etf_short_names = ['USO','UNG','DBC','DBA','GLD','USCI']
        currency_etf_short_names = ['FXY','FXE','FXB','FXF','FXC','FXA']
        print('fetching sp constituents ...')
        if self.use_datahub:
            sp = list(pd.read_csv(url_constituents).Symbol)
        else:
            sp = list(pd.read_csv('./sp_constituents.csv').Symbol)
        ret = sp + spydr_short_names + commodity_etf_short_names + currency_etf_short_names
        return ret
    
    
    def build_pg_from_csvs(self,delete_table_before_building=False):
        try:
            # always try to build the table in case it's the first time
            sql = f"""
            create table {self.full_table_name}(
                symbol text not null,
                date Date not null,
                open numeric not null,
                high numeric not null,
                low numeric not null,
                close numeric not null,
                adj_close numeric not null,
                volume integer not null,
                primary key(symbol,Date));
            """            
            self.pga.exec_sql_raw(sql) 
        except:
            # ignore
            pass
        stk_files = [s+'.csv' for s in self.initial_symbol_list] if self.initial_symbol_list is not None else   [f for f in listdir(self.STOCKS_DIR) if isfile(join(self.STOCKS_DIR, f))] 
        for csv_name in stk_files:
            csv_path = f'{self.STOCKS_DIR}/{csv_name}'
            try:
                df = pd.read_csv(csv_path)
                sym = csv_name.replace('.csv','')            
                self.write_symbol_to_pg(sym,df)
            except Exception as e:
                self.logger.warn(str(e))
    
    def add_symbol_to_pg(self,symbol,dt_beg,dt_end):
        df = self.get_yahoo_data(symbol,dt_beg,dt_end)
        if df is None or len(df)<1:
            self.logger.warn(f'add_symbol_to_pg ERROR: no data retrieved for symbol {symbol}')
            return
        self.write_symbol_to_pg(symbol, df)

    def write_symbol_to_pg(self,symbol,df):
        if len(df)>0:
            df['symbol'] = symbol
            df_this_stock = self.yahoo_to_pg(df)
            df_already_there = self.pga.get_sql(f"select date from {self.full_table_name} where symbol='{symbol}'")
            df_to_write = df_this_stock.copy()
            if len(df_already_there) > 0:
                df_to_write = df_this_stock[~df_this_stock.date.isin(df_already_there.date)]
            if len(df_to_write)<1:
                self.logger.warn(f'write_symbol_to_pg: no new data to write for symbol {symbol}')
                return
            self.logger.info(f'writing {symbol} to database')
            self.pga.write_df_to_postgres_using_metadata(df=df_to_write,table_name=self.full_table_name)
        else:
            raise ValueError(f'cannot find Yahoo data for {symbol}')        
    
    def update_daily_with_delete(self,dt_beg=None,dt_end=None):
        '''
        Update existing symbols in database by deleting data between beg and end dates first\
        :param dt_beg:
        :param dt_end:
        '''
        pga2 = self.pga
        end_date = dt_end if dt_end is not None else dt.datetime.now()
        end_date_str = end_date.strftime("%Y-%m-%d")
        beg_date = dt_beg if dt_beg is not None else end_date - dt.timedelta(self.days_to_fetch)
        beg_date_str = beg_date.strftime("%Y-%m-%d")
        sql_delete = f"""
        delete from {self.full_table_name} where date>='{beg_date_str}' and date<='{end_date_str}';
        """
        pga2.exec_sql_raw(sql_delete)
        self.update_yahoo_daily(dt_beg=dt_beg,dt_end=dt_end)
    
    def update_yahoo_daily(self,dt_beg=None,dt_end=None):
        '''
        Update existing symbols in database with new days data
        :param dt_beg:
        :param dt_end:
        '''
        pga2 = self.pga
#         end_date = dt_end if dt_end is not None else dt.datetime.now()
#         end_date_str = end_date.strftime("%Y-%m-%d")
#         beg_date = dt_beg if dt_beg is not None else end_date - dt.timedelta(self.days_to_fetch)
#         beg_date_str = beg_date.strftime("%Y-%m-%d")
#         sql_delete = f"""
#         delete from {self.full_table_name} where date>='{beg_date_str}' and date<='{end_date_str}';
#         """
#         pga2.exec_sql_raw(sql_delete)
#                 
        sql_get = f"""
        select symbol,max(date) max_date, min(date) min_date from {self.full_table_name}
        group by symbol
        """
                
        df_last_dates = pga2.get_sql(sql_get).sort_values('symbol')
        total_to_update = len(df_last_dates)
        for i in range(len(df_last_dates)):
            r  = df_last_dates.iloc[i]
            
            end_date = dt_end if dt_end is not None else dt.datetime.now()
            end_date = get_last_business_day(end_date)
            end_date_morn = dt.datetime(int(end_date.year),int(end_date.month),int(end_date.day),1,0)                
            beg_date = dt_beg if dt_beg is not None else end_date_morn - dt.timedelta(self.days_to_fetch)
            db_min_date = dt.datetime.combine(r.min_date, dt.datetime.min.time())
            db_max_date = dt.datetime.combine(r.max_date, dt.datetime.max.time())
            
            if (db_min_date - beg_date).days <= 4: # account for weekends + or long holiday
                # move the begin date up because you already have this data
                db_max_date_morn = dt.datetime(int(db_max_date.year),int(db_max_date.month),int(db_max_date.day),1,0)
                beg_date = db_max_date_morn + dt.timedelta(1)   
            if beg_date >= end_date:
                self.logger.info(f'{r.symbol} number {i} of {total_to_update}  nothing to update')
                continue   
            if end_date <= db_max_date:
                self.logger.info(f'{r.symbol} number {i} of {total_to_update}  nothing to update')
                continue                   
            try:
                self.add_symbol_to_pg(r.symbol, beg_date, end_date)
                self.logger.info(f'{r.symbol} number {i} of {total_to_update}  updated')
            except Exception as e:
                self.logger.warn(str(e))
                continue
            

    def get_pg_data(self,symbol,dt_beg,dt_end):
        sql_dt_beg = dt_beg.strftime('%Y-%m-%d')
        sql_dt_end = dt_end.strftime('%Y-%m-%d')
        
        sql_get = f"""
        select * from {self.full_table_name}
        where symbol='{symbol}' and date>='{sql_dt_beg}' and date<='{sql_dt_end}'
        """        
        df = self.pga.get_sql(sql_get)
#         df = self.pg_to_yahoo(df)
        return df
    
    def yahoo_to_pg(self,df_in):
        df = df_in.copy()
        df = df.rename(columns = {c:c.lower().replace(' ','_') for c in df.columns.values})
        if 'date' not in df.columns.values and df.index.name.lower()=='date':
            df['date'] = df.index
            df.index = range(len(df))
        return df
    
    def pg_to_yahoo(self,df_in):
        df = df_in.copy()
        df.index = df.date
        df = df.rename(columns = {c:c[0].upper()+c[1:].replace('_',' ') for c in df.columns.values})
        df = df.rename(columns={'Adj close':'Adj Close'})
        return df        
    
        
        
    def delete_pg_table(self):
        self.pga.exec_sql_raw(f"drop table if exists {self.full_table_name}")
        sql = f"""
        create table {self.full_table_name}(
            symbol text not null,
            date Date not null,
            open numeric not null,
            high numeric not null,
            low numeric not null,
            close numeric not null,
            adj_close numeric not null,
            volume integer not null,
            primary key(symbol,Date));
        """            
        self.pga.exec_sql_raw(sql) 
    
    def execute(self):
        if self.delete_table:
            self.delete_pg_table()

        if self.delete_schema:
            self.pga.exec_sql_raw(f"DROP SCHEMA IF EXISTS  {self.schema_name};")
            self.pga.exec_sql_raw(f"create schema {self.schema_name};")
        
            
        if self.fetch_from_yahoo:
            hist_dict = self.build_history_dict()
            self.write_hist_dict_to_csv(hist_dict=hist_dict)
        if self.build_table:
            self.build_pg_from_csvs()
        if self.update_table:
            self.update_yahoo_daily(self.beg_date, self.end_date)
#             self.update_daily_with_delete(self.beg_date, self.end_date)


if __name__ == '__main__':
    logger = li.init_root_logger('logger.log','INFO') 
    start_time = dt.datetime.now()
    logger.info(f'starting at {start_time}')
    parser = ap.ArgumentParser()
#     parser.add_argument('--action',type=str,help='update (default), build_from_scratch, build_from_csvs, add_new_symbols',
#                         default='update')


    parser.add_argument('--delete_schema',type=bool,
                    help='delete schema (default=False)',
                    default=False)
    parser.add_argument('--delete_table',type=bool,
                    help='delete_table schema (default=False)',
                    default=False)
    parser.add_argument('--fetch_from_yahoo',type=bool,
                    help='fetch_from_yahoo schema (default=False)',
                    default=False)
    parser.add_argument('--build_table',type=bool,
                    help='build_table schema (default=False)',
                    default=False)
    parser.add_argument('--update_table',type=bool,
                    help='update_table data (default=False)',
                    default=False)
    parser.add_argument('--beg_date_yyyymmddhhmmss',type=str,
                    help='yyyymmdd or yyyymmddhhmmss string that is converted to beginning datetime.dateime object for yahoo fetches (default datetime.datetime.now - datetime.timedelta(days_to_fetch)',
                    nargs='?')
    parser.add_argument('--end_date_yyyymmddhhmmss',type=str,
                    help='yyyymmdd or yyyymmddhhmmss string that is converted to ending datetime.dateime object for yahoo fetches (default datetime.datetime.now)',
                    nargs='?')
    parser.add_argument('--dburl',type=str,
                    help='database url (None will be localhost)',
                    nargs='?')
    parser.add_argument('--databasename',type=str,
                    help='databasename (None will be maindb)',
                    nargs='?')
    parser.add_argument('--username',type=str,
                    help='username (None will be postgres)',
                    nargs='?')
    parser.add_argument('--password',type=str,
                    help='password (None will be blank)',
                    nargs='?')
    parser.add_argument('--schema_name',type=str,
                    help='schema name for table (None will be test_schema)',
                    nargs='?')
    parser.add_argument('--yahoo_daily_table',type=str,
                    help='table name for table (None will be yahoo_daily)',
                    nargs='?')
    parser.add_argument('--initial_symbol_list',type=str,
                    help='comma separated list of symbols, like SPY,AAPL,XLE (default is list of SP500 stocks and main sector and commodity etfs)',
                    nargs='?')
    parser.add_argument('--days_to_fetch',type=int,
                    help=f"number of days of history to fetch (None will be {DEFAULT_DAYS_TO_FETCH})",
                    default=DEFAULT_DAYS_TO_FETCH)
    args = parser.parse_args()

    days_to_fetch = args.days_to_fetch
    nw = dt.datetime.now()
    end_date = dt.datetime(int(nw.year),int(nw.month),int(nw.day),23,59) 
    if args.end_date_yyyymmddhhmmss is not None:
        yyyy = args.end_date_yyyymmddhhmmss[0:4]
        month = args.end_date_yyyymmddhhmmss[4:6]
        day = args.end_date_yyyymmddhhmmss[6:8]
        #dt.datetime.max.time()        
        hour = args.end_date_yyyymmddhhmmss[8:10] if len(args.end_date_yyyymmddhhmmss)>8 else 23
        minute = args.end_date_yyyymmddhhmmss[10:12] if len(args.end_date_yyyymmddhhmmss)>10 else 1
        second =  args.end_date_yyyymmddhhmmss[12:14] if len(args.end_date_yyyymmddhhmmss)>12 else 1
        end_date = dt.datetime(yyyy,month,day,hour,minute,second)
    end_date_morn = dt.datetime(int(end_date.year),int(end_date.month),int(end_date.day),1,0)
    beg_date = end_date_morn  - dt.timedelta(days_to_fetch) 
    
    if args.beg_date_yyyymmddhhmmss is not None:
        yyyy = args.beg_date_yyyymmddhhmmss[0:4]
        month = args.beg_date_yyyymmddhhmmss[4:6]
        day = args.beg_date_yyyymmddhhmmss[6:8]
        #dt.datetime.max.time()        
        hour = args.beg_date_yyyymmddhhmmss[8:10] if len(args.beg_date_yyyymmddhhmmss)>8 else 23
        minute = args.beg_date_yyyymmddhhmmss[10:12] if len(args.beg_date_yyyymmddhhmmss)>10 else 1
        second =  args.beg_date_yyyymmddhhmmss[12:14] if len(args.beg_date_yyyymmddhhmmss)>12 else 1
        beg_date = dt.datetime(yyyy,month,day,hour,minute,second)

    hb = HistoryBuilder(
        delete_schema=args.delete_schema, 
        delete_table=args.delete_table, 
        fetch_from_yahoo=args.fetch_from_yahoo, 
        build_table=args.build_table, 
        update_table=args.update_table, 
        beg_date=beg_date, 
        end_date=end_date, 
        dburl=args.dburl, 
        databasename=args.databasename, 
        username=args.username, 
        password=args.password, 
        schema_name=args.schema_name, 
        yahoo_daily_table=args.yahoo_daily_table, 
        initial_symbol_list=args.initial_symbol_list, 
        days_to_fetch=args.days_to_fetch, 
        logger=logger)
    
    hb.execute()
    end_time = dt.datetime.now()
    logger.info(f'ending at {end_time}')
    elapsed_time = end_time - start_time
    logger.info(f'elapsed time {elapsed_time}')

    
