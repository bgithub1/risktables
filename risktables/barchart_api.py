'''
Created on Dec 12, 2018

implement the barchart python api: https://www.barchart.com/ondemand/api/getHistory
@author: Bill Perlman
'''
import ondemand
import pandas as pd
import datetime
import numpy as np

class BcHist():
    codes = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}    
    month_nums = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}
    
    endpoints = {
                 'free_url':'https://marketdata.websol.barchart.com/',
                 'paid_url':'http://ondemand.websol.barchart.com/'}
    
    def get_month_nums(self,month_code):
        '''
        get dictionary of month code to month number
        '''
        month_num = {BcHist.codes[num]:num for num in BcHist.codes.keys()}[month_code]
        return int(month_num)
    
    def __init__(self,
                 api_key,
                 bar_type='minutes',
                 interval= 1,
                 endpoint_type=None):
        self.api_key = api_key
        self.bar_type = bar_type
        self.interval = interval
        self.bars_per_day = 60*24 if bar_type != 'daily'  else 1 
        endpoint_to_use = BcHist.endpoints['free_url'] if endpoint_type is None else BcHist.endpoints[endpoint_type]
        self.od =  ondemand.OnDemandClient(api_key=api_key, end_point=endpoint_to_use)

    
    def get_history(self,short_name,beg_yyyymmdd=None,end_yyyymmdd=None):
        date_dict = calc_dates(beg_yyyymmdd, end_yyyymmdd, self.bar_type, self.interval)
        resp = self.od.history(short_name, self.bar_type, 
                          maxRecords=date_dict['max_records'], 
                          interval=self.interval,
                          startDate=date_dict['beg_yyyymmdd'])
        status = resp['status']
        results = resp['results']
        df = pd.DataFrame(results) 
        return (status,df)  

    def get_contract_short_names(self,commodity,beg_yyyymm,end_yyyymm,contract_month_list=None):
        # define commodity month code list
        
        # get beg and end yy and mm
        beg_yy = int(str(beg_yyyymm)[2:4])
        beg_mm = int(str(beg_yyyymm)[4:6])
        end_yy = int(str(end_yyyymm)[2:4])
        end_mm = int(str(end_yyyymm)[4:6])
        yys = np.linspace(beg_yy,end_yy,end_yy-beg_yy + 1) 
        curr_mm = beg_mm
        contracts = []
        yyyymms = []
        for yy in yys:
            #Step 1: create months for this year
            end_month = 12 if yy<end_yy else end_mm
            months = contract_month_list
            if months is None:
                months = np.linspace(curr_mm, end_month, end_month - curr_mm + 1)
            #Step 2: create yyyymm's for thisyear
            these_yyyymms = ['%04d%02d' %(yy+2000,i) for i in months]
            yyyymms.extend(these_yyyymms)

            #Step 3: create contracts for this year
            month_codes = [BcHist.codes[i] for i in months]
            these_mcmms = ['%s%02d' %(m,yy) for m in month_codes]
            these_contracts = ['%s%s' %(commodity,mcmm) for mcmm in these_mcmms]
            contracts.extend(these_contracts)
            
            # reset current month to 1
            curr_mm = 1
        df = pd.DataFrame({'yyyymm':yyyymms,'contract':contracts})
        
        return df

        
    def get_last_trading_day(self,contract):
        month_code = str(contract)[-3:][0]
        prev_month_num = self.get_month_nums(month_code) -1
        yyyy = int(str(contract)[-2:])+2000
        if prev_month_num < 1:
            prev_month_num = 12
            yyyy -=1
        yyyymmdd = '%04d%02d01' %(yyyy,prev_month_num)
        resp = self.od.history(contract, 'daily', 
                          maxRecords=60, 
                          interval=self.interval,
                          startDate=yyyymmdd)
        
        results = resp['results']
        if results is None:
            raise ValueError(resp['status'])
        df = pd.DataFrame(results) 
        last_row = df.iloc[len(df)-1]
        return int(last_row.tradingDay.replace('-',''))
    
    
def calc_dates(beg_yyyymmdd=None,end_yyyymmdd=None,
                     bar_type='minutes',interval=1):
    if beg_yyyymmdd is not None:            
        ds = str(beg_yyyymmdd)
        byyyy = int(ds[0:4])
        bmm = int(ds[4:6])
        bdd = int(ds[6:8])
    if end_yyyymmdd is not None:
        ds = str(end_yyyymmdd)
        eyyyy = int(ds[0:4])
        emm = int(ds[4:6])
        edd = int(ds[6:8])

    if beg_yyyymmdd is None:
        if end_yyyymmdd is not None:
            dt_end = datetime.datetime(eyyyy,emm,edd)
            dt_beg = dt_end - datetime.timedelta(40)
        else:
            dt_end = datetime.datetime.now()
            dt_beg = dt_end - datetime.timedelta(40)
    else:
        dt_beg = datetime.datetime(byyyy,bmm,bdd)
        if end_yyyymmdd is not None:
            dt_end = datetime.datetime(eyyyy,emm,edd)
        else:
            dt_end = dt_beg + datetime.timedelta(40)
        
    
    diff = dt_end - dt_beg
    days = diff.days
    max_records  = 1 * days 
    if bar_type != 'daily':
        max_records = 24*60 * days // interval
    b_yyyymmdd = '%04d%02d%02d' %(dt_beg.year,dt_beg.month,dt_beg.day)
    e_yyyymmdd = '%04d%02d%02d' %(dt_end.year,dt_end.month,dt_end.day)
    return {'max_records':max_records,'beg_yyyymmdd':b_yyyymmdd,'end_yyyymmdd':e_yyyymmdd}


if __name__=='__main__':
    api_key = open('temp_folder/free_api_key.txt','r').read()
    bc = BcHist(api_key, 'daily', 1, 'free_url')
    t = datetime.datetime.now() - datetime.timedelta(30)
    y = t.year - 2000
    m = t.month
    mcode = BcHist.codes[m]
    contract = 'CL%s%02d' %(mcode,y)
    print('last trading day (based on history) for %s = %d' %(contract,bc.get_last_trading_day(contract)))
    pass
             
