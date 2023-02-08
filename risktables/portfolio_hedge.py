'''
Created on Mar 1, 2019

@author: bperlman1
'''
import sys,os
from passlib.utils.compat import num_types
if  not './' in sys.path:
    sys.path.append('./')
if  not '../' in sys.path:
    sys.path.append('../')

import pandas as pd

import numpy as np
# import os,sys
# import pdb
import torch 
from torch import nn
from torch.autograd import Variable
from torch import optim
# import torch.nn.functional as F
from risktables import var_models as vm
import datetime
import matplotlib.pyplot as plt
from textwrap import wrap
import argparse as ap
from itertools import combinations



RANDOM_PORTFOLIO_PATH = './temp_folder/df_random_portfolio.csv'
RANDOM_PORTFOLIO_WEGHTS = './temp_folder/random_portfolio_weights.csv'
SPDR_HISTORY_PATH = './temp_folder/df_hist_portfolio_hedge.csv'

def fetch_histories(symbol_list,dt_beg=None,dt_end=None):
    yf = vm.YahooFetcher()
    dt_end = dt_end if dt_end is not None else datetime.datetime.now()
    dt_beg = dt_beg if dt_beg is not None else dt_end - datetime.timedelta(30*5)
    yf.fetch_histories(symbol_list, dt_beg, dt_end)
    histories = yf.history_dict
    close_dict = {symbol:list(histories[symbol].close) for symbol in histories.keys()}
    df_hist = pd.DataFrame(close_dict)
    return df_hist

def create_random_portfolio_history(num_of_symbols=20,weights=None,dt_beg=None,dt_end=None,csv_save_path=None):
    url_constituents = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    df_constit = pd.read_csv(url_constituents)
    all_symbols = sorted(list(df_constit.Symbol))
    random_indices = np.random.randint(0,len(all_symbols)-1,num_of_symbols)
    symbol_list = [all_symbols[i] for i in random_indices]
    w = weights
    if w is None:
        w = np.random.rand(len(symbol_list))
    port_path = RANDOM_PORTFOLIO_PATH if csv_save_path is None else csv_save_path
    df =  create_portfolio_history(symbol_list,weights=w,dt_beg=dt_beg,dt_end=dt_end)
    df_spdr = fetch_sector_spdr_df(refresh=True)
    df_spdr = df_spdr.drop('SPY',axis=1)
    df_spdr['port'] = df.port
    df_spdr.to_csv(port_path,index=None)
    df_random_port_weights = pd.DataFrame({'symbol':symbol_list,'position':w})
    df_random_port_weights.to_csv(RANDOM_PORTFOLIO_WEGHTS,index=False)
    return df_spdr

def create_portfolio_history(symbol_list,weights,dt_beg=None,dt_end=None):
    '''
    Given a list of symbols in symbol_list, create a set of portfolio values, where 
    each portolio value is the weighted sum of each day's closing prices for the securites in 
    symbol_list.  

    The returned DataFrame will have a single column, named "port", for each day between
    dt_beg and dt_end, where the value of port will be the weighted sum for that day.
    
    The length of symbol_list MUST equal the length of weights.
    The sum of weightw must = 1

    :param symbol_list: list of SP 500 stocks
    :param weights: list of values adding up to 1
    :param dt_beg: default is 150
    :param dt_end: default is today
    '''
    df_hist = fetch_histories(symbol_list,dt_beg,dt_end)
#     hist_matrix = df_hist[symbol_list].as_matrix()
    hist_matrix = df_hist[symbol_list].values
    # now create random weights
    prices = hist_matrix @ weights
    df = pd.DataFrame({'port':prices})
    return df

def fetch_sector_spdr_df(refresh=False,csv_save_path=None):
    hist_path = SPDR_HISTORY_PATH if csv_save_path is None else csv_save_path
    if refresh:
        symbol_list = ['SPY','XLE', 'XLU', 'XLK', 'XLB', 'XLP', 'XLY', 'XLI', 'XLC', 'XLV', 'XLF']
        df_hist = fetch_histories(symbol_list)
        df_hist.to_csv(hist_path,index=None)
    else:
        df_hist = pd.read_csv(hist_path)
    return df_hist  



# main model class
class SingleLayerNet(nn.Module):
    def __init__(self, D_in, D_out):
        super(SingleLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, D_out) 
    def forward(self, x):
        return self.linear1(x)


class PortfolioHedge():
    def __init__(self,df,portfolio_value_col,date_column=None,num_of_test_days=None):
        '''        
        :param df: pandas DataFrame containing historical prices for each security that you will use to hedge,
            and the prices of your portfolio in a column whose name = portfolio_value_col.
            If df == None, then this class will use the sector spdr ETFs as the hedging securities
        :param portfolio_value_col: the name of the column in df which holds the hitorical prices of your portfolio.
            IF None, then use 'SPY' has your portfolio.
        :param date_column: None ,if your DataFrame does not have a date column, otherwise the column name of that column
        :param num_of_test_days: Number or rows in df to use as out of sample data. If None, then use int(len(df) * .1).
            The size of the training set will equal len(df) - num_of_test_days
        '''
        self.portfolio_value_col = portfolio_value_col
        self.df = df
        self.date_column  = date_column
        ntd = num_of_test_days
        if ntd is None:
            ntd = int(len(self.df) * .1)
        self.num_of_test_days = ntd
    
    def create_last_day_ratio(self,hedge_ratio_dict):
        '''
        Create ratio between the simulated last training day price, and the actual training portfolio price

        '''
        df_train = self.df.iloc[:-self.num_of_test_days]        
        non_port_columns = sorted(list(filter(lambda c: c != self.portfolio_value_col,self.df.columns.values)))
        last_day_hedge_price_vector = np.array(df_train[non_port_columns].iloc[-1])
        hedge_ratio_vector = [hedge_ratio_dict[k] for k in non_port_columns]
        last_day_simulated_price = last_day_hedge_price_vector @ hedge_ratio_vector
        last_day_real_port = df_train[self.portfolio_value_col].iloc[-1]
        last_day_ratio = last_day_real_port / last_day_simulated_price
        return last_day_ratio
    
    def get_train_test_values(self):
        df = self.df.copy()
        ntd = self.num_of_test_days
        yreal = df[self.portfolio_value_col].values.reshape(-1)
        df = df.drop(self.portfolio_value_col,axis=1)
        if self.date_column is not None:
            df = df.drop(self.date_column)
        all_Xnp = df.values.reshape(-1,len(df.columns.values))
        hedge_ratios = np.array([self.hedge_ratio_dict[symbol] for symbol in df.columns.values])
        ysim = np.array(all_Xnp @ hedge_ratios + self.bias) * self.last_day_ratio
        # plot with without pandas
        x_train = list(range(len(all_Xnp)))[:-ntd]
        x_test =  list(range(len(all_Xnp)))[-ntd-1:]
        ysim_train = ysim[:-ntd]
        ysim_test = ysim[-ntd-1:] 
        yreal_train = yreal[:-ntd]
        yreal_test = yreal[-ntd-1:]
        ret_dict = {'x_train':x_train,'x_test':x_test,
                'ysim_train':ysim_train,'ysim_test':ysim_test,
                'yreal_train':yreal_train,'yreal_test':yreal_test
        }
        return ret_dict
        
    def plot_hedge_ratios_vs_real(self):
        d = self.get_train_test_values()
        x_train = d['x_train']
        x_test = d['x_test']
        ysim_train = d['ysim_train']
        ysim_test = d['ysim_test']
        yreal_train = d['yreal_train']
        yreal_test = d['yreal_test']
        fig, ax = plt.subplots(figsize = (16,7))
    
        ax.plot(x_train,yreal_train,color='blue',label='y_train_real')
        ax.plot(x_train,ysim_train,color='orange',label='y_train_model')
        ax.plot(x_test,yreal_test,color='red',label='y_test_real')
        ax.plot(x_test,ysim_test,color='green',label='y_test_model')
        ax.legend()
        ax.grid()
        hr = {k:round(self.hedge_ratio_dict[k],4) for k in self.hedge_ratio_dict.keys()}
        t = f'{self.portfolio_value_col} vs {hr}'
        t = t.replace("'","")
        title = ax.set_title("\n".join(wrap(t, 60)))
        fig.tight_layout()
        title.set_y(1.05)
        fig.subplots_adjust(top=0.8)
        plt.show()


class MinVarianceHedge(PortfolioHedge):
    def __init__(self,df,portfolio_value_col,date_column=None,num_of_test_days=None):
        '''        
        :param df: pandas DataFrame containing historical prices for each security that you will use to hedge,
            and the prices of your portfolio in a column whose name = portfolio_value_col.
            If df == None, then this class will use the sector spdr ETFs as the hedging securities
        :param portfolio_value_col: the name of the column in df which holds the hitorical prices of your portfolio.
            IF None, then use 'SPY' has your portfolio.
        :param date_column: None ,if your DataFrame does not have a date column, otherwise the column name of that column
        :param num_of_test_days: Number or rows in df to use as out of sample data. If None, then use int(len(df) * .1).
            The size of the training set will equal len(df) - num_of_test_days
        '''
        super(MinVarianceHedge,self).__init__(df=df,portfolio_value_col=portfolio_value_col,
                                          date_column=date_column,num_of_test_days=num_of_test_days)

    def run_model(self):
        non_port_columns = sorted(list(filter(lambda c: c != self.portfolio_value_col,self.df.columns.values)))
        all_columns = [self.portfolio_value_col] + non_port_columns
        df_train = self.df[all_columns].iloc[:-self.num_of_test_days]
        df_corr = df_train.corr()
#         matrix_corr_inner = df_corr.as_matrix()[1:,1:]
        matrix_corr_inner = df_corr.values[1:,1:]
        matrix_inverse = np.linalg.inv(matrix_corr_inner)
        non_port_vector = np.array(df_corr.iloc[1:,0])
        hedges = matrix_inverse @ non_port_vector
        self.hedge_ratio_dict = {non_port_columns[i]:hedges[i] for i in range(len(non_port_columns))}
        self.bias = 0
#         last_day_hedge_price_vector = np.array(df_train[non_port_columns][-1])
#         hedge_ratio_vector = [self.hedge_ratio_dict[k] for k in non_port_columns]
#         last_day_simulated_price = last_day_hedge_price_vector @ hedge_ratio_vector
#         last_day_real_port = df_train[self.portfolio_value_col][-1]
#         last_day_ratio = last_day_real_port / last_day_simulated_price
        self.last_day_ratio = self.create_last_day_ratio(self.hedge_ratio_dict)

        

class PytorchHedge(PortfolioHedge):
    '''
    Create hedge rations using a simple pytorch Linear model.
    
    Toy Example where your portfolio is SPY, and you want to hedge it using the sector spdr's:
    ph = PytorchHedge()
    ph.run_model()
    ph.plot_hedge_ratios_vs_real()
    print(ph.hedge_ratio_dict)

    Example of a 20 random memebers of the SP 500 as your portfolio, with random weights, and the sector spdr's as your hedge
    yf = 
    '''
    def __init__(self,df,portfolio_value_col,date_column=None,num_of_test_days=None):
        '''        
        :param df: pandas DataFrame containing historical prices for each security that you will use to hedge,
            and the prices of your portfolio in a column whose name = portfolio_value_col.
            If df == None, then this class will use the sector spdr ETFs as the hedging securities
        :param portfolio_value_col: the name of the column in df which holds the hitorical prices of your portfolio.
            IF None, then use 'SPY' has your portfolio.
        :param date_column: None ,if your DataFrame does not have a date column, otherwise the column name of that column
        :param num_of_test_days: Number or rows in df to use as out of sample data. If None, then use int(len(df) * .1).
            The size of the training set will equal len(df) - num_of_test_days
        '''
        super(PytorchHedge,self).__init__(df=df,portfolio_value_col=portfolio_value_col,
                                          date_column=date_column,num_of_test_days=num_of_test_days)
        

    
    
    def run_model(self):
#         Ynp = self.df[self.portfolio_value_col].as_matrix()[:-self.num_of_test_days]
        Ynp = self.df[self.portfolio_value_col].values[:-self.num_of_test_days]
        x_cols = list(filter(lambda s: s.lower() != self.portfolio_value_col.lower(),self.df.columns.values))
        if self.date_column is not None:
            x_cols = list(filter(lambda s: s.lower()!= self.date_column.lower(),x_cols))
#         Xnp = self.df[x_cols].as_matrix()[:-self.num_of_test_days]
        Xnp = self.df[x_cols].values[:-self.num_of_test_days]
        b=1
        # number of epochs
        epochs=20000
        # instantiate model
        m1 = SingleLayerNet(Xnp.shape[1],1)
        # Create input torch Variables for X and Y
        X = Variable(torch.Tensor(Xnp))
        Y = Variable(torch.Tensor(Ynp).reshape(-1,1))
        
        # create loss and optimize
        
        loss_fn = nn.MSELoss(size_average = False) 
        optimizer = optim.Adam(m1.parameters(), lr = 0.01)
        
        # Training loop
        for i in range(epochs):
            # create a batch of x values and y values (labels)
            indices = list(range(Xnp.shape[0]))
            np.random.shuffle(indices)
            xb = X[indices[:b]]    
            yb = Y[indices][:b]
            # zero the optimizer
            optimizer.zero_grad()  # clear previous gradients
            
            # execute the forward pass to compute y values from equation xA^T + b (the linear transformation)
            output_batch = m1(xb)           # compute model output
            
            # calculate a loss
            loss = loss_fn(output_batch, yb)  # calculate loss
            # compute gradients
            loss.backward()        # compute gradients of all variables wrt loss
            optimizer.step()       # perform updates using calculated gradients
            # print out progress
            if i % 500 == 0 :
                if loss.data < .5:
                    break
                print('epoch {}, loss {}'.format(i,loss.data))
        
        # print model results
        model_A = m1.linear1.weight.data.numpy()
        model_bias = m1.linear1.bias.data.numpy()
        self.hedge_ratio_dict = {x_cols[i]:model_A[0][i] for i in range(len(x_cols))}
        self.bias = model_bias[0]
        self.last_day_ratio = self.create_last_day_ratio(self.hedge_ratio_dict)

def best_hedge(df,portfolio_column_name='port',max_hedge_symbols=4,rounding_value_for_hedge_comarisons=.001):
    hedge_cols = df.columns.values
    hedge_cols = np.setdiff1d(hedge_cols, np.array([portfolio_column_name]))

    
    sets = []
    for i in range(1,max_hedge_symbols+1):
        for l in combinations(hedge_cols,i): 
            sets.append(l)
    l
    lowest_diff = sys.float_info.max
    best_set = None
    best_ph = None
    for s in sets:
        dfs = df[[portfolio_column_name] + list(s)]
        ph = MinVarianceHedge(dfs,portfolio_column_name)
        ph.run_model()
        d = ph.get_train_test_values()
        first_ysim_test = d['ysim_test'][1]
        first_yreal_test = d['yreal_test'][1]
        abs_diff = abs(first_yreal_test-first_ysim_test)
        # round it
        update = False
        if best_ph is None:
            update=True
        elif abs_diff < lowest_diff:
            if abs(abs_diff/lowest_diff - 1) < rounding_value_for_hedge_comarisons:
                if len(s) < len(best_set):
                    update=True
            else:
                update=True
        if update:
            lowest_diff = abs_diff
            best_set = s
            best_ph = ph
                    
    return best_ph

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--use_min_variance',type=bool,
                        help='Use minimum variance calculation, as opposed to Pytorch regression. (Default = False)',
                        default=False)
    parser.add_argument('--use_spy',type=bool,
                        help='Use SPY as your portfolio, otherwise use 20 randomly created members of SP 500, with random weights. (Default = False)',
                        default=False)
    parser.add_argument('--refetch_data',type=bool,
                        help='Re-fetch all data. (Default = False)',
                        default=False)
    args = parser.parse_args()

    use_min_variance = args.use_min_variance
    use_spy = args.use_spy
    refetch_data = args.refetch_data
    
    if use_spy:
        portfolio_column_name = 'SPY'
        df = fetch_sector_spdr_df(refresh=refetch_data)
    else:
        portfolio_column_name = 'port'
        if refetch_data:
            df = create_random_portfolio_history()
        else:
            df = pd.read_csv(RANDOM_PORTFOLIO_PATH)

    if use_min_variance:
        ph = MinVarianceHedge(df,portfolio_column_name)
    else:
        ph = PytorchHedge(df,portfolio_column_name)
    ph.run_model()
    ph.plot_hedge_ratios_vs_real()
    print(ph.hedge_ratio_dict)
