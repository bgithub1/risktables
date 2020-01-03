'''
Created on Jan 3, 2020

@author: bperlman1
'''
import sys,os

if  not os.path.abspath('./') in sys.path:
    sys.path.append(os.path.abspath('./'))
if  not os.path.abspath('../') in sys.path:
    sys.path.append(os.path.abspath('../'))
import dash
import dash_html_components as html


import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas_datareader.data as pdr
import datetime
import pytz
import pathlib
from risktables import pg_pandas as pg
from dateutil.relativedelta import *

from risktables import dgrid_components as dgc
import argparse as ap


CONTRACTS_TO_DISPLAY_DICT = {'names':['E-Mini SP','Nymex Crude','Ice Brent','NYMEX Natural Gas'], 
                             'symbols':['ES','CL','CB','NG']
}                             

# Create css styles for some parts of the display

STYLE_TITLE={
    'line-height': '20px',
    'textAlign': 'center',
    'background-color':'#47bacc',
    'color':'#FFFFF9',
    'vertical-align':'middle',
} 

STYLE_UPGRID = STYLE_TITLE.copy()
STYLE_UPGRID['background-color'] = '#EAEDED'
STYLE_UPGRID['line-height'] = '10px'
STYLE_UPGRID['color'] = '#21618C'
STYLE_UPGRID['height'] = '50px'

MONTH_CODES = 'FGHJKMNQUVXZ'
DICT_MONTH_CODE = {MONTH_CODES[i]:i+1 for i in range(len(MONTH_CODES))}

OUTPUT_COLS = ['contractSymbol','strike','mid','dte_pct','ert',
                'forward_price','deltak','p1','p2','p3','e1','e2','e3']

DICT_THRESHOLD_BIDS = {'ES':1,'CL':.02,'CB':.02,'NG':.002}
DICT_DELTAK = {'ES':5,'CL':.5,'CB':.5,'NG':.1}

ALL_SYMBOLS = None # filled in from __main__ at run time
ALL_PRODUCTS = None # filled in from __main__ at run time


from pandas.tseries.holiday import USFederalHolidayCalendar
bday_us = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
TIMEZONE = 'US/Eastern'

def get_rate_df(num_months_for_rate=1,start_datetime=None,end_datetime=None):
    # see if you can get Libor from the FRED API
    if start_datetime is None:
        n = datetime.datetime.now() - 252*bday_us #datetime.timedelta(7)
    else:
        if type(start_datetime)==int:
            sdt = start_datetime
            n = datetime.datetime(int(str(sdt)[0:4]),int(str(sdt)[4:6]), int(str(sdt)[6:8]))
        else:
            n = start_datetime 
    y = n.year
    m = n.month
    d = n.day
    beg = '%04d-%02d-%02d' %(y,m,d)
    ed = end_datetime if end_datetime is not None else (datetime.datetime.now() - datetime.timedelta(1))
    if type(ed)==int:
        ed = datetime.datetime(int(str(ed)[0:4]), int(str(ed)[4:6]), int(str(ed)[6:8]))
    y = ed.year
    m = ed.month
    d = ed.day
    eds = '%04d-%02d-%02d' %(y,m,d)
    fred_libor_file = f'USD{num_months_for_rate}MTD156N'
    df = pdr.DataReader(fred_libor_file, "fred", f'{beg}', f'{eds}')
    if len(df)<1:
        raise ValueError(f'FRED calendar of {fred_libor_file} does not contain dates {beg} through {eds}')
    df['fixed_rate'] = df[f'USD{num_months_for_rate}MTD156N'].astype(float)/100
    df.columns = df.columns.get_level_values(0)
    df['date_yyyymmdd'] = [int(d.year*100*100) + int(d.month*100) + int(d.day) for d in df.index]
    df.index = range(len(df))
    df['prate'] = df.shift(1).fixed_rate
    df['fixed_rate'] = df.apply(lambda r:r.prate if r.fixed_rate !=r.fixed_rate else r.fixed_rate,axis=1)
    return df[['date_yyyymmdd','fixed_rate']]


def get_rate(num_months_for_rate=1,rate_datetime=None):
    # see if you can get Libor from the FRED API
    if rate_datetime is None:
        n = datetime.datetime.now() - 7*bday_us #datetime.timedelta(7)
    else:
        n = rate_datetime #- datetime.timedelta(14)
    y = n.year
    m = n.month
    d = n.day
    beg = '%04d-%02d-%02d' %(y,m,d)
#     ed = n  + datetime.timedelta(1)
    ed = datetime.datetime.now()
    y = ed.year
    m = ed.month
    d = ed.day
    eds = '%04d-%02d-%02d' %(y,m,d)
    fred_libor_file = f'USD{num_months_for_rate}MTD156N'
    df = pdr.DataReader(fred_libor_file, "fred", f'{beg}', f'{eds}')
    if len(df)<1:
        raise ValueError(f'FRED calendar of {fred_libor_file} does not contain dates {beg} through {eds}')
#     fixed_rate = float(df.iloc[len(df)-1][f'USD{num_months_for_rate}MTD156N'])/100
    fixed_rate = float(df.iloc[-1][f'USD{num_months_for_rate}MTD156N'])/100
    return fixed_rate

def get_nth_weekday(year,month,target_weekday,nth_occurrence):
    '''
    weekday is the term that assigns numbers from 0 to 6 to the days of the weeks.
    weekday 0 = monday
    '''
    # get dayofweeks of year,month,1
    weekday_01 = datetime.datetime(year,month,1).weekday()
    if weekday_01 <= target_weekday:
        day_of_month_of_first_occurence = target_weekday - weekday_01
        day_of_month_of_nth_occurence = day_of_month_of_first_occurence + 1 + (nth_occurrence - 1) * 7
    else:
        day_of_month_of_nth_occurence = target_weekday - weekday_01 + 1 + (nth_occurrence) * 7 
    return datetime.datetime(year,month,day_of_month_of_nth_occurence)




def get_ES_expiry(symbol):
    monthcode_yy = symbol[2:]
    month = DICT_MONTH_CODE[monthcode_yy[0]]
    year = 2000 + int(monthcode_yy[1:])
    return get_nth_weekday(year,month,4,3)

def get_E6_expiry(symbol):
    monthcode_yy = symbol[2:]
    next_month = DICT_MONTH_CODE[monthcode_yy[0]] + 1
    year = 2000 + int(monthcode_yy[1:])
    if next_month>12:
        next_month = 1
        year += 1
    return datetime.datetime(year,next_month,1) - 7*bday_us

def get_CL_expiry(symbol):
    monthcode_yy = symbol[2:]
    month = DICT_MONTH_CODE[monthcode_yy[0]]
    year = 2000 + int(monthcode_yy[1:])
    month = month -1
    if month<1:
        month = 12
        year = year - 1
    return datetime.datetime(year,month,26) - 7*bday_us

def get_NG_expiry(symbol):
    monthcode_yy = symbol[2:]
    month = DICT_MONTH_CODE[monthcode_yy[0]]
    year = 2000 + int(monthcode_yy[1:])
    return datetime.datetime(year,month,1) - 4*bday_us

DICT_PRODUCT = {
    'E6':get_E6_expiry,
    'ES':get_ES_expiry,
    'CL':get_CL_expiry,
    'NG':get_NG_expiry,
}

    
def get_expiry(symbol):
    product = symbol[:2]
    f = DICT_PRODUCT[product]
    return f(symbol)


def dt_from_yyyymmdd(yyyymmdd,hour=0,minute=0,timezone=TIMEZONE):
    y = int(str(yyyymmdd)[0:4])
    m = int(str(yyyymmdd)[4:6])
    d = int(str(yyyymmdd)[6:8])  
    return datetime.datetime(y,m,d,hour,minute,tzinfo=pytz.timezone(timezone))

def yyyymmdd_from_dt(dt):
    y = int(dt.year)
    m = int(dt.month)
    d = int(dt.day)
    return y*100*100 + m*100 + d

def get_dte_pct(trade_yyyymmdd,expiry_yyyymmdd):
    dt_td = dt_from_yyyymmdd(trade_yyyymmdd)
    dt_xp = dt_from_yyyymmdd(expiry_yyyymmdd)
    return ((dt_xp - dt_td).days + 1)/365


def get_all_years_per_product(product):
    global ALL_SYMBOLS
    return sorted(list(set([s[3:] for s in ALL_SYMBOLS if s[:2] == product])))
def get_all_monthcodes_per_product(product,year):
    global ALL_SYMBOLS
    yy = str(year)[-2:]
    return sorted(list(set([s[2] for s in ALL_SYMBOLS if (s[:2] == product) & (s[-2:]==yy)])))
def get_dates_per_symbol(symbol):
    sql_dates = f'''
    select min(settle_date) min_date, max(settle_date) max_date 
    from sec_schema.options_table 
    where symbol='{symbol}'; 
    '''
    df_dates_per_symbol = pga.get_sql(sql_dates)
    min_date_yyyymmdd = int(df_dates_per_symbol.iloc[0].min_date)
    max_date_yyyymmdd = int(df_dates_per_symbol.iloc[0].max_date)
    # subtract 11 days from max_date_yyyymmdd
    max_date_yyyymmdd = yyyymmdd_from_dt(dt_from_yyyymmdd(max_date_yyyymmdd) - datetime.timedelta(11))
    return [min_date_yyyymmdd,max_date_yyyymmdd]

def get_all_symbols(pga):
    ALL_SYMBOL_SQL = 'select distinct symbol from sec_schema.options_table;'
    ALL_SYMBOLS = pga.get_sql(ALL_SYMBOL_SQL).symbol.values
    ALL_PRODUCTS = sorted(list(set([s[:2] for s in ALL_SYMBOLS])))
    return (ALL_SYMBOLS,ALL_PRODUCTS)

dict_fut_mon = {'F':'H','G':'H','H':'H','J':'M','K':'M','M':'M','N':'U','Q':'U','U':'U','V':'Z','X':'Z','Z':'Z'}
def get_valid_series_from_barchartacs(symbol,trade_date_yyyymmdd,
                                      interest_rate=None):
    threshold_bid = DICT_THRESHOLD_BIDS[symbol[:2]]
    deltak = DICT_DELTAK[symbol[:2]]
    
#     print(symbol,trade_date_yyyymmdd)
    ir = interest_rate 
    if ir is None:
        ir = get_rate(1,dt_from_yyyymmdd(trade_date_yyyymmdd))
        
#     # ****** Step 01: get futures data from sql
    fm = dict_fut_mon[symbol[2]]
    futures_symbol = symbol[0:2] + fm + symbol[3:]
    sql = f'''
    select * from sec_schema.underlying_table ot
    where ot.symbol='{futures_symbol}'  and settle_date={trade_date_yyyymmdd}
    ;
    '''
    df_futures = pga.get_sql(sql)
    forward_price = df_futures.iloc[0].close

#     # ****** Step 02: get options data from sql
    sql = f'''
    select * from sec_schema.options_table ot
    where ot.symbol='{symbol}'  and settle_date={trade_date_yyyymmdd}
    ;
    '''
    df_options = pga.get_sql(sql)
    dfc = df_options[(df_options.pc.str.lower()=='c')][['settle_date','strike','close','pc']].sort_values('strike')
    dfc = dfc[dfc.strike >= forward_price]
    dfp = df_options[(df_options.pc.str.lower()=='p')][['settle_date','strike','close','pc']].sort_values('strike')
    dfp = dfp[dfp.strike < forward_price]
    atm_strike = dfc[dfc.strike == dfc.strike.min()].iloc[0].strike
    
    # ******  Step 03: merge puts and alls into one dataframe
    dfb = dfc.append(dfp).sort_values(['pc','strike'])    
    dfb.pc = dfb.pc.str.lower()
    dfb = dfb.rename(columns={'pc':'cp','settle_date':'trade_date_yyyymmdd','close':'mid'})
    dfb = dfb[dfb.mid >= threshold_bid ]
    exp_dt  = get_expiry(symbol)
    expiry_yyyymmdd = int(exp_dt.year)*100*100 + int(exp_dt.month)*100 + int(exp_dt.day)
    dfb['expiry_yyyymmdd'] = expiry_yyyymmdd
    df_ret = dfb.copy()
    df_ret['trade_date_yyyymmdd'] = trade_date_yyyymmdd
    df_ret['forward_price'] = forward_price
    
    df_ret = df_ret.sort_values(['cp','strike'])
    df_ret.index = range(len(df_ret))

    # ******  Step 04: Add m dte_pct, ert  
    dte_pct = get_dte_pct(trade_date_yyyymmdd,expiry_yyyymmdd)
    df_ret['dte_pct'] = dte_pct
    ert = np.exp(dte_pct * ir)
    df_ret['ert'] = ert
    
    df_ret['k_over_fp'] = df_ret.strike / forward_price
    df_ret['deltak'] = deltak
    
    # ****** Step 05: create p1,p2,p3, e1,e2,e3 unit values
    df_ret['p1']= df_ret.apply(lambda r: r.deltak / r.strike**2 * r.mid,axis=1)
    df_ret['p2'] = df_ret.apply(lambda r: 2 * r.p1 *(1-np.log(r.k_over_fp)),axis=1)
    df_ret['p3'] = df_ret.apply(lambda r: 3 * r.p1 * (2*np.log(r.k_over_fp) - np.log(r.k_over_fp)**2),axis=1)
    e1 = -(1+np.log(forward_price/atm_strike) - forward_price/atm_strike)
    e2 = 2 * np.log(atm_strike/forward_price) * (forward_price/atm_strike - 1) + 1/2 * np.log(atm_strike/forward_price)**2
    e3 = 3 * np.log(atm_strike/forward_price)**2 * (1/3 * np.log(atm_strike/forward_price) - 1 + forward_price/atm_strike)
    df_ret['e1'] = e1
    df_ret['e2'] = e2
    df_ret['e3'] = e3
    return df_ret

def create_skew(df_series):
    e1 = df_series.iloc[0].e1
    e2 = df_series.iloc[0].e2
    e3 = df_series.iloc[0].e3
    ert = df_series.iloc[0].ert
    p1 = ert * -1 * df_series.p1.sum() + e1
    p2 = ert * df_series.p2.sum() + e2
    p3 = ert * df_series.p3.sum() + e3
    S_ =  (p3 - 3*p1*p2 + 2*p1**3) / (p2 - p1**2)**(3/2)
    SKEW = 100 - 10*S_
    return SKEW

def create_skew_df(symbol,beg_yyyymmdd, end_yyyymmdd,deltak=5):
    trade_date_yyyymmdd_beg = beg_yyyymmdd
    trade_date_yyyymmdd_end = end_yyyymmdd
    df_rates = get_rate_df(1,trade_date_yyyymmdd_beg,trade_date_yyyymmdd_end)
    futures_symbol = symbol[:(len(symbol)-3)] + dict_fut_mon[symbol[2]] + symbol[3:]
    sql = f'''
    select * from sec_schema.underlying_table ot
    where ot.symbol='{futures_symbol}'  and settle_date>={trade_date_yyyymmdd_beg} and settle_date <= {trade_date_yyyymmdd_end}
    ;
    '''
    trade_dates_yyyymmdd = pga.get_sql(sql).settle_date.values
    dates = []
    final_skews  = []
    dict_df_exp_pair = {}
    for d in trade_dates_yyyymmdd:
        ir = df_rates[df_rates.date_yyyymmdd==d].iloc[0].fixed_rate
        dfes = get_valid_series_from_barchartacs(symbol,d,interest_rate=ir)
        final_skew = create_skew(dfes)
        dates.append(d)
        final_skews.append(final_skew)
        dict_df_exp_pair[d] = {
            'df_spx_1':dfes.copy(),
                               'final_skew':final_skew
        }
    df_pga_skew = pd.DataFrame({'date':dates,'final_skew':final_skews})    
    return df_pga_skew

def show_data(sym):
    global DICT_DELTAK
    # get all dates for this sym
    beg_end_yyyymmdds = get_dates_per_symbol(sym)
    # get last date
    end_yyyymmdd = beg_end_yyyymmdds[1]
    # make first date 60 days back from last date
    dt_beg = dt_from_yyyymmdd(end_yyyymmdd) - datetime.timedelta(60)
    beg_yyyymmdd = yyyymmdd_from_dt(dt_beg)
    prod = sym[:2]
    deltak = DICT_DELTAK[prod]
    dfs = create_skew_df(sym,beg_yyyymmdd,end_yyyymmdd,deltak=deltak)
    return dfs



if __name__=='__main__':
    
    df_pg_info = pd.read_csv('./postgres_info.csv')
    postgres_config_names = ' '.join(df_pg_info.config_name.values)
    parser = ap.ArgumentParser()
    parser.add_argument('--host',type=str,default='127.0.0.1',help='host/ip address of server')
    parser.add_argument('--port',type=int,default=8600,help='port of server')

    parser.add_argument('--database_config_name',type=str,
                        help=f'IF not specified, do not use postgres.  If used, one of {postgres_config_names}',
                        default='secdb_local')
    parser.add_argument('--additional_route',type=str,nargs='?',
                        help='the additional URI, if needed (like /oilgas or /risk if the full URL has to include it')
    args = parser.parse_args()
    config_name = args.database_config_name
    
    df_this_config = df_pg_info[df_pg_info.config_name==config_name].fillna('')
    if len(df_this_config)<1:
        raise ValueError(f'postgres configuration name {config_name} is not in ')
    s = df_this_config.to_dict('records')[0]

    pga = pg.PgPandas(
                dburl=s['dburl'], databasename=s['databasename'], 
                username=s['username'], password=s['password'])
    ALL_SYMBOLS, ALL_PRODUCTS = get_all_symbols(pga)



    logger = dgc.init_root_logger('logfile.log','WARN') 
    
    
    top_div = html.Div([
                        dgc.dcc.Markdown('''
                        # Commodity Option SKEW Analysis
                        Select a Commodity, Year and Monthcode below. The resulting data is derived from the CBOE SKEW formula outlined in the whitepaper: 
                        
                        (https://www.cboe.com/micro/skew/documents/skewwhitepaperjan2011.pdf)
                        '''
                        ,style={'color':'white'})
                ],
                style=STYLE_TITLE,id='top_div')
    
    
    dropdown_instructions = dgc.DivComponent('dd_instructions',initial_children=['Select from the Product, Year and Month Dropdowns'])
    
    chained_dd_prods = dgc.ChainedDropDownDiv('chained_dd_prods',
                    initial_dropdown_labels=['Emini','WTI Crude','Brent Crude'],
                    initial_dropdown_values=['ES','CL','CB'])
    
    def _chained_years(inputs):
        prod = inputs[1]
        if prod is None or len(prod)<1:
            return []
        yys = get_all_years_per_product(prod)
        choices = [{'label':str(2000 + int(yy)),'value':yy} for yy in yys]
        return  choices
    
        
    chained_dd_years = dgc.ChainedDropDownDiv('chained_dd_years',
                    dropdown_input_components=[chained_dd_prods],
                    choices_transformer_method=_chained_years,
                    placeholder="Select a year")
    
    def _chained_months(inputs):
        if inputs is None or len(inputs)<3:
            return []
        prod = inputs[1]
        if prod is None or len(prod)<1:
            return []
        yy = inputs[2]
        if yy is None or len(yy)<1:
            return []
        year = 2000 + int(yy)
        mcs = get_all_monthcodes_per_product(prod,year)
        choices = [{'label':mc ,'value':mc} for mc in mcs]
        return  choices
    
        
    chained_dd_months = dgc.ChainedDropDownDiv('chained_dd_months',
                    dropdown_input_components=[chained_dd_prods,chained_dd_years],
                    choices_transformer_method=_chained_months,
                    placeholder="Select a month code")
    
    full_symbol_store_inputs = [
        (chained_dd_prods.dropdown_id,'value'),
        (chained_dd_years.dropdown_id,'value'),
        (chained_dd_months.dropdown_id,'value'),    
    ]
    
    def _create_full_symbol(inputs):
        print(f'_create_full_symbol inputs {inputs}')
        if inputs is None or len(inputs)<3 or inputs[0] is None or inputs[1] is None or inputs[2] is None:
            return {}
        prod = inputs[0]
        yy = str(inputs[1])[-2:]
        month = inputs[2]
        full_symbol = prod+month+yy
        full_symbol = full_symbol.upper()
        print(f'full_symbol {full_symbol}')
        dict_df = show_data(full_symbol).to_dict()
        return {'full_symbol':full_symbol,'df':dict_df}
        
    full_symbol_store = dgc.StoreComponent('symbol_store',full_symbol_store_inputs,
                                create_data_dictionary_from_df_transformer=_create_full_symbol)
    
    def _symbol_from_store(inputs):
        print(f'_symbol_from_store inputs: {inputs}')
        if inputs is None or len(inputs)<1:
            return ['']
        sym_dict = inputs[0]
        if sym_dict is None or len(sym_dict)<1:
            return ['']
        return [sym_dict['full_symbol']]
    
    full_symbol_div = dgc.DivComponent('full_symbol_div',input_component=full_symbol_store,
                                      callback_input_transformer=_symbol_from_store)
    
    def transform_input_to_df(dict_df,key_of_df,columns_to_show=None):
        df = None
        dict_this_risk = None
        if len(dict_df)>0:
            dict_this_risk = dict_df[key_of_df]
            df = dgc.make_df(dict_this_risk)
            if columns_to_show is not None:
                df = df[columns_to_show]
        return df
    
    
    dash_graph = dgc.XyGraphComponent('dash_graph',full_symbol_store,'date',
                    title="Skew Graph",plot_bars=False,
                    transform_input=lambda dict_df: transform_input_to_df(dict_df,'df'))
    
    dash_table = dgc.DashTableComponent('dash_table',None,input_component=full_symbol_store,
                    title="Skew Data",
                    transform_input=lambda dict_df: transform_input_to_df(dict_df,'df'),
                    columns_to_round=[],digits_to_round=3)
    

    app_to_use = dash.Dash(url_base_pathname='/skew/')
    # app.layout = html.Div(children=[chained_dd.html])
    
    app_component_list = [top_div,dropdown_instructions,chained_dd_prods,
                          chained_dd_years,chained_dd_months,full_symbol_store,full_symbol_div,dash_graph,dash_table]
    
    gtcl = ['1fr','4fr 1fr 1fr 1fr','0fr 1fr','1fr','1fr']
    app = dgc.make_app(app_component_list,
                    app=app_to_use,
                    grid_template_columns_list=gtcl)    
    
    
    # Step 5: run the server    
    host = args.host
    port = args.port
    app.run_server(host=host,port=port)
    
