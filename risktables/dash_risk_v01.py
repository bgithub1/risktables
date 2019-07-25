'''
Created on Jul 23, 2019

Create a dash app to view risk statistics about a portfolio that you upload from your local
file system.

The portfolio should have 2 colunns:
    symbol:  Any valid Yahoo finance symbol like SPY, AAPL, XLE
    position:  a signed integer or decimal number that represents the number units per price that you have bought or sold. 

@author: bperlman1
'''
import sys,os
from risktables.risk_tables import DEFAULT_PORTFOLIO_NAME
paths_to_add_to_sys_path = ['./','../','../../dashgrid','../../dashgrid/dashgrid']
for p in paths_to_add_to_sys_path:
    if  not p in sys.path:
        sys.path.append(os.path.abspath(p))
    
from risktables import risk_tables
from dashgrid import dgrid_components as dgc
import dash_html_components as html
import argparse as ap
import pandas as pd
from IPython import display

STYLE_TITLE={
    'line-height': '20px',
#     'borderWidth': '1px',
#     'borderStyle': 'none',
#     'borderRadius': '1px',
    'textAlign': 'center',
    'background-color':'#21618C',
    'color':'#FFFFF9',
    'vertical-align':'middle',
} 

DEFAULT_PORTFOLIO_NAME=  './spdr_stocks.csv'

def create_risk_summary_divs(logger,store_all_risk_dfs_comp):
    def _value_from_risk_dict(input_list,risk_key):
        risk_value =  input_list[0][risk_key]
        risk_value = round(float(str(risk_value)),2)
        risk_text = f'{risk_key}: {risk_value}'
        return [risk_text]
    
    var_1_99_div = dgc.DivComponent('var_1_99', 
                input_component=store_all_risk_dfs_comp,
                callback_input_transformer = lambda input_list:_value_from_risk_dict(input_list, 'port_var'), 
                logger=logger)
    sp_equiv_div = dgc.DivComponent('sp_equiv', 
                input_component=store_all_risk_dfs_comp,
                callback_input_transformer = lambda input_list:_value_from_risk_dict(input_list, 'sp_dollar_equiv'), 
                logger=logger)

    port_delta_div = dgc.DivComponent('port_delta', 
                input_component=store_all_risk_dfs_comp,
                callback_input_transformer = lambda input_list:_value_from_risk_dict(input_list, 'delta'), 
                logger=logger)
    port_gamma_div = dgc.DivComponent('port_gamma', 
                input_component=store_all_risk_dfs_comp,
                callback_input_transformer = lambda input_list:_value_from_risk_dict(input_list, 'gamma'), 
                logger=logger)
    port_vega_div = dgc.DivComponent('port_vega', 
                input_component=store_all_risk_dfs_comp,
                callback_input_transformer = lambda input_list:_value_from_risk_dict(input_list, 'vega'), 
                logger=logger)
    port_theta_div = dgc.DivComponent('port_theta',
                input_component=store_all_risk_dfs_comp,
                callback_input_transformer = lambda input_list:_value_from_risk_dict(input_list, 'theta'), 
                logger=logger)
    return [var_1_99_div,sp_equiv_div,port_delta_div,port_gamma_div,port_vega_div,port_theta_div]
    
def create_risk_dts(logger,store_all_risk_dfs_comp):
    dt_risk1_comp = dgc.DashTableComponent('risk1',None,store_all_risk_dfs_comp,
                title='Risk by Symbol',
                transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_risk_all'),
                logger=logger,columns_to_round=[],digits_to_round=3)

    dt_risk2_comp = dgc.DashTableComponent('risk2',None,store_all_risk_dfs_comp,
                title='Risk by Underlying',
                transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_risk_by_underlying'),
                logger=logger,columns_to_round=[],digits_to_round=3)

    dt_corr_returns = dgc.DashTableComponent('corr_returns',None,store_all_risk_dfs_comp,
                title='Returns Correlations',
                transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_corr'),
                logger=logger,columns_to_round=[],digits_to_round=4)

    dt_corr_price = dgc.DashTableComponent('corr_price',None,store_all_risk_dfs_comp,
                title='Price Correlations',
                transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_corr_price'),
                logger=logger,columns_to_round=[],digits_to_round=4)

    df_atm_info = dgc.DashTableComponent('atm_info',None,store_all_risk_dfs_comp,
                title='Atm, Std and N-day High-Low',
                transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_atm_info'),
                logger=logger,columns_to_round=[],digits_to_round=2)
    
    return [dt_risk1_comp,dt_risk2_comp,dt_corr_returns,dt_corr_price,df_atm_info]

def risk_data_closure(use_postgres=False,
                             dburl='localhost',
                             databasename='testdb',
                             username='',
                             password='',
                             schema_name='test_schema',
                             yahoo_daily_table='yahoo_daily'):
    '''
    This method returns a method to access historical data from either yahoo, or a postgres db.
    :param use_postgres:
    :param dburl:
    :param databasename:
    :param username:
    :param password:
    :param schema_name:
    :param yahoo_daily_table:
    '''
    def create_risk_data(input_list):
        if len(input_list)<1:
            return None
        dict_df = input_list[0]
        df_portfolio = dgc.make_df(dict_df)
        rt = risk_tables.RiskCalcs(
            use_postgres=use_postgres, 
            dburl=dburl, databasename=databasename, 
            username=username, password=password, 
            schema_name=schema_name, yahoo_daily_table=yahoo_daily_table)
        
        # Step 3: get an actual portfolio
        risk_dict_dfs = rt.calculate(df_portfolio)
        return risk_dict_dfs
    return create_risk_data
    
def transform_risk_input_to_df(risk_dict,key_of_df,columns_to_show=None):
    if len(risk_dict)<1:
        return None
    dict_df_var = risk_dict[key_of_df]
    df = dgc.make_df(dict_df_var)
    if columns_to_show is not None:
        df = df[columns_to_show]
    return df

def dash_app(create_risk_data_method,
             dash_app_logger=None):
    '''
    
    :param create_risk_data_method: a method that has one input, which is a list.  See the closure method risk_data_closure
                                to see how this method gets created
    :param dash_app_logger:
    '''
    # create a logger
    logger = dgc.li.init_root_logger('logfile.log','INFO') if dash_app_logger is None else dash_app_logger
    
    top_div = html.Div([html.H3('Portfolio Risk Analysis'),
                html.H4(' (See Quick Start at page bottom for help)')],
                style=STYLE_TITLE)
    
    u1_comp = dgc.UploadComponent('u1',
                'choose a csv portfolio file with symbol and position columns',
                logger=logger)
    
    h1_comp = dgc.UploadFileNameDiv('h1',u1_comp)

    df_init = pd.read_csv(DEFAULT_PORTFOLIO_NAME)    

    dt1_comp = dgc.DashTableComponent('dt1',df_init,u1_comp,
                title='Main Portfolio',
                editable_columns=['position'],
                logger=logger)
    
    logger = dt1_comp.logger
#     dict_risk = create_risk_data_method([df_init.to_dict()])
#     df_risk_all = dgc.make_df(dict_risk['df_risk_all'])
#     df_risk_by_underlying = dgc.make_df(dict_risk['df_risk_by_underlying'])
    
    store_all_risk_dfs_comp = dgc.StoreComponent('store_all_risk_dfs', dt1_comp, 
            create_data_dictionary_from_df_transformer=create_risk_data_method, logger=logger)
    
        
    gr1_comp = dgc.XyGraphComponent('gr1',store_all_risk_dfs_comp,
                x_column='symbol',
                title='VaR Per Underlying Security',
                transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_var',['symbol','position']),
                logger=logger)    

#     cols_to_round_dfra = [c for c in df_risk_all.columns.values if c not in ['symbol','underlying','position']]
#     dt_risk1_comp = dgc.DashTableComponent('risk1',df_risk_all,store_all_risk_dfs_comp,
#                 title='Risk by Symbol',
#                 transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_risk_all',df_risk_all.columns.values),
#                 logger=logger,columns_to_round=cols_to_round_dfra,digits_to_round=3)
#  
#     cols_to_round_dfru = [c for c in df_risk_all.columns.values if c not in ['symbol','underlying','position']]
#     dt_risk2_comp = dgc.DashTableComponent('risk2',df_risk_by_underlying,store_all_risk_dfs_comp,
#                 title='Risk by Underlying',
#                 transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_risk_by_underlying',df_risk_by_underlying.columns.values),
#                 logger=logger,columns_to_round=cols_to_round_dfru,digits_to_round=3)
#  
#     cols_to_round_corr_ret = [] # round all number columns
#     dt_corr_returns = dgc.DashTableComponent('corr_returns',None,store_all_risk_dfs_comp,
#                 title='Returns Correlations',
#                 transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_corr'),
#                 logger=logger,columns_to_round=cols_to_round_corr_ret,digits_to_round=4)
#  
#     dt_corr_price = dgc.DashTableComponent('corr_price',None,store_all_risk_dfs_comp,
#                 title='Price Correlations',
#                 transform_input=lambda risk_dict: transform_risk_input_to_df(risk_dict,'df_corr_price'),
#                 logger=logger,columns_to_round=cols_to_round_corr_ret,digits_to_round=4)

    # create as list of all components that can be put in layout
#     app_component_list = [top_div,u1_comp,h1_comp,store_all_risk_dfs_comp,dt1_comp,gr1_comp,dt_risk1_comp,dt_risk2_comp,dt_corr_returns,dt_corr_price]
    risk_summparies = create_risk_summary_divs(logger,store_all_risk_dfs_comp)
    risk_comps = create_risk_dts(logger,store_all_risk_dfs_comp)
    app_component_list = [top_div,u1_comp,h1_comp,store_all_risk_dfs_comp] + risk_summparies + [dt1_comp,gr1_comp] + risk_comps
    gtcl = ['1fr','50% 50%','1fr','50% 50%','25% 25% 25% 25%','50% 50%','50% 50%','50% 50%','100%']
    
    app = dgc.make_app(app_component_list,grid_template_columns_list=gtcl)
    return app

if __name__=='__main__':
    df_pg_info = pd.read_csv('./postgres_info.csv')
    postgres_config_names = ' '.join(df_pg_info.config_name.values)
    parser = ap.ArgumentParser()
    parser.add_argument('--host',type=str,default='127.0.0.1',help='host/ip address of server')
    parser.add_argument('--port',type=int,default=8500,help='port of server')
    parser.add_argument('--initial_portolio',type=str,default=DEFAULT_PORTFOLIO_NAME,
                        help='initial portfolio to Load')
    parser.add_argument('--database_config_name',type=str,nargs='?',
                        choices=postgres_config_names,
                        help=f'IF not specified, do not use postgres.  If used, one of {postgres_config_names}')

    args = parser.parse_args()
    config_name = args.database_config_name
    
    if config_name is  None:
        risk_data_callback = risk_data_closure()
    else:
        df_this_config = df_pg_info[df_pg_info.config_name==config_name].fillna('')
        if len(df_this_config)<1:
            raise ValueError(f'postgres configuration name {config_name} is not in ')
        s = df_this_config.to_dict('records')[0]
        create_risk_data_method = risk_data_closure(use_postgres=True, 
                    dburl=s['dburl'], databasename=s['databasename'], 
                    username=s['username'], password=s['password'], 
                    schema_name=s['schema_name'], 
                    yahoo_daily_table=s['table_names'])
    app = dash_app(create_risk_data_method)
    app.run_server(host=args.host,port=args.port)


    
    
    