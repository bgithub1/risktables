'''
Created on Jul 21, 2019

Define classes that inherit dgrid.ComponentWrapper.  
1. These classes facilitate use of:
    dash_core_components
    dash_html_components
2. They free the developer from having to implement their own callbacks
     on dash_core_components instances.
3. They make it easy to place dash_core_components in a flexible grid.

@author: bperlman1
'''

import datetime,base64,io,pytz

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output#,State
from dash.exceptions import PreventUpdate
import dash_table
import pandas as pd
import numpy as np
from plotly.offline import  iplot
import traceback
import re
import plotly.graph_objs as go
from plotly.graph_objs.layout import Margin#,Font
import dash
import flask
import logging

def init_root_logger(logfile='logfile.log',logging_level='INFO'):
    level = logging_level
    if level is None:
        level = logging.DEBUG
    # get root level logger
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        return logger
    logger.setLevel(logging.getLevelName(level))

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)   
    return logger

DEFAULT_LOG_PATH = './logfile.log'
DEFAULT_LOG_LEVEL = 'DEBUG'

#**************************************************************************************************


grid_style = {'display': 'grid',
              'border': '1px solid #000',
              'grid-gap': '8px 8px',
              'background-color':'#fffff9',
            'grid-template-columns': '1fr 1fr'}

file_download_grid_style = {'display': 'grid',
              'border': '1px solid #000',
              'grid-gap': '8px 8px',
              'background-color':'#ffffff',
            'grid-template-columns': '1fr 1fr'}

chart_style = {'margin-right':'auto' ,'margin-left':'auto' ,'height': '98%','width':'98%'}

#**************************************************************************************************

#*********************** define useful css styles ***********************
borderline = 'none' #'solid'
button_style={
    'line-height': '40px',
    'borderWidth': '1px',
    'borderStyle': borderline,
    'borderRadius': '1px',
    'textAlign': 'center',
    'background-color':'#fffff0',
    'vertical-align':'middle',
}
button_style_no_border={
    'line-height': '40px',
    'textAlign': 'center',
    'background-color':'#fffff0',
    'vertical-align':'middle',
}

blue_button_style={
    'line-height': '40px',
#     'borderWidth': '1px',
#     'borderStyle': borderline,
#     'borderRadius': '1px',
    'textAlign': 'center',
    'background-color':'#A9D0F5',#ed4e4e
    'vertical-align':'middle',
}

border_style={
    'line-height': '40px',
    'border':borderline + ' #000',
    'textAlign': 'center',
    'vertical-align':'middle',
}

table_like = {
    'display':'table',
    'width': '100%'
}

# define h4_like because h4 does not play well with dash_table
h4_like = { 
    'display': 'table-cell',
    'textAlign' : 'center',
    'vertical-align' : 'middle',
    'font-size' : '16px',
    'font-weight': 'bold',
    'width': '100%',
    'color':'#22aaff'
}

DEFAULT_TIMEZONE = 'US/Eastern'

# ****************************************** Define logic to facilitate creation of grids *****************************************

class GridItem():
    def __init__(self,child,html_id=None):
        self.child = child
        self.html_id = html_id
    @property
    def html(self):
        if self.html_id is not None:
            return html.Div(children=self.child,className='grid-item',id=self.html_id)
        else:
            return html.Div(children=self.child,className='grid-item')

def create_grid(component_array,num_columns=2,column_width_percents=None,additional_grid_properties_dict=None,
                wrap_in_loading_state=False):
    gs = grid_style.copy()
    percents = [str(round(100/num_columns-.001,1))+'%' for _ in range(num_columns)] if column_width_percents is None else [str(c)+'%' for c in column_width_percents]
    perc_string = " ".join(percents)
    gs['grid-template-columns'] = perc_string 
    if additional_grid_properties_dict is not None:
        for k in additional_grid_properties_dict.keys():
            gs[k] = additional_grid_properties_dict[k]           
#     g =  html.Div([GridItem(c).html if type(c)==str else c.html for c in component_array], style=gs)

    div_children = []
    for c in component_array:
        if type(c)==str:
            div_children.append(GridItem(c).html)
        elif hasattr(c,'html'):
            div_children.append(c.html)
        else:
            div_children.append(c)
    if wrap_in_loading_state:
        g = dcc.Loading(html.Div(div_children,style=gs),type='cube')
    else:
        g = html.Div(div_children,style=gs)
    return g



# ************************* define useful factory methods *****************

def parse_contents(contents):
    '''
    app.layout contains a dash_core_component object (dcc.Store(id='df_memory')), 
      that holds the last DataFrame that has been displayed. 
      This method turns the contents of that dash_core_component.Store object into
      a DataFrame.
      
    :param contents: the contents of dash_core_component.Store with id = 'df_memory'
    :returns pandas DataFrame of those contents
    '''
    c = contents.split(",")[1]
    c_decoded = base64.b64decode(c)
    c_sio = io.StringIO(c_decoded.decode('utf-8'))
    df = pd.read_csv(c_sio)
    # create a date column if there is not one, and there is a timestamp column instead
    cols = df.columns.values
    cols_lower = [c.lower() for c in cols] 
    if 'date' not in cols_lower and 'timestamp' in cols_lower:
        date_col_index = cols_lower.index('timestamp')
        # make date column
        def _extract_dt(t):
            y = int(t[0:4])
            mon = int(t[5:7])
            day = int(t[8:10])
            hour = int(t[11:13])
            minute = int(t[14:16])
            return datetime.datetime(y,mon,day,hour,minute,tzinfo=pytz.timezone(DEFAULT_TIMEZONE))
        # create date
        df['date'] = df.iloc[:,date_col_index].apply(_extract_dt)
    return df

def make_df(dict_df):
    if type(dict_df)==list:
        if type(dict_df[0])==list:
            dict_df = dict_df[0]
        return pd.DataFrame(dict_df,columns=dict_df[0].keys())
    else:
        return pd.DataFrame(dict_df,columns=dict_df.keys())

class BadColumnsException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)
    
def create_dt_div(dtable_id,df_in=None,
                  columns_to_display=None,
                  editable_columns_in=None,
                  title='Dash Table',logger=None,
                  title_style=None):
    '''
    Create an instance of dash_table.DataTable, wrapped in an dash_html_components.Div
    
    :param dtable_id: The id for your DataTable
    :param df_in:     The pandas DataFrame that is the source of your DataTable (Default = None)
                        If None, then the DashTable will be created without any data, and await for its
                        data from a dash_html_components or dash_core_components instance.
    :param columns_to_display:    A list of column names which are in df_in.  (Default = None)
                                    If None, then the DashTable will display all columns in the DataFrame that
                                    it receives via df_in or via a callback.  However, the column
                                    order that is displayed can only be guaranteed using this parameter.
    :param editable_columns_in:    A list of column names that contain "modifiable" cells. ( Default = None)
    :param title:    The title of the DataFrame.  (Default = Dash Table)
    :param logger:
    :param title_style: The css style of the title. Default is dgrid_components.h4_like.
    '''
    # create logger 
    lg = init_root_logger() if logger is None else logger
    
    lg.debug(f'{dtable_id} entering create_dt_div')
    
    # create list that 
    editable_columns = [] if editable_columns_in is None else editable_columns_in
    datatable_id = dtable_id
    dt = dash_table.DataTable(
        page_current= 0,
        page_size= 100,
        filter_action='none', # 'fe',
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left',
            } for c in ['symbol', 'underlying']
        ],

        style_as_list_view=False,
        style_table={
            'maxHeight':'450px','overflowX': 'scroll','overflowY':'scroll'
#             'height':'15','overflowX': 'scroll','overflowY':'scroll'
        } ,
        editable=True,
        css=[{"selector": "table", "rule": "width: 100%;"}],
        id=datatable_id
    )
    if df_in is None:
        df = pd.DataFrame({'no_data':[]})
    else:
        df = df_in.copy()
        if columns_to_display is not None:
            if any([c not in df.columns.values for c in columns_to_display]):
                m = f'{columns_to_display} are missing from input data. Your input Csv'
                raise BadColumnsException(m)           
            df = df[columns_to_display]
            
    dt.data=df.to_dict('rows')
    dt.columns=[{"name": i, "id": i,'editable': True if i in editable_columns else False} for i in df.columns.values]                    
    s = h4_like if title_style is None else title_style
    child_div = html.Div([html.Div(html.Div(title,style=s),style=table_like),dt])
    lg.debug(f'{dtable_id} exiting create_dt_div')
    return child_div

def file_store_transformer(contents):
    '''
    Convert the contents of the file that results from use of dash_core_components.Upload class.  
        This method gets called from the UploadComponent class callback.
    :param contents:    The value received from an update of a dash_core_components.Upload instance.'
                            
    '''
    if contents is None or len(contents)<=0 or contents[0] is None:
        d =  None
    else:
        d = parse_contents(contents).to_dict('rows')
    return d


def dataframe_rounder(df,digits=2,columns_to_round=None):
    if df is None or len(df)<1:
        return df
    if 'dataframe' not in str(type(df)).lower():
        raise ValueError('dataframe_rounder EXCEPTION: input df is NOT a Dataframe')
    if columns_to_round is not None:
        isok = False
        if type(columns_to_round)==list:
            isok=True
        if type(columns_to_round)==set:
            isok=True
        if type(columns_to_round)==np.array:
            isok=True
        if type(columns_to_round)==tuple:
            isok=True
        if not isok:
            raise ValueError('dataframe_rounder EXCEPTION: columns_to_round type is not a list, set, np.array or tuple') 
    cols = columns_to_round
    if cols is None or len(cols)<1:
        cols = []        
        # get first row
        df1row = df.iloc[0:1]
        for c in df1row.columns.values:
            v = df1row.iloc[0][c]
            v = str(v).strip()
            r = re.findall('^[e\-0-9\.]{1,}$',v)
            if len(r)>0:
                cols.append(c)
    df_ret = df.copy()
    for c in cols:
        df_ret[c] = df_ret[c].round(digits)
    return df_ret
    

def plotly_plot(df_in,x_column,plot_title=None,
                y_left_label=None,y_right_label=None,
                bar_plot=True,figsize=(16,10),
                number_of_ticks_display=20,
                marker_color=None,
                yaxis2_cols=None):
    '''
    Create a plotly.graph_objs.Graph figure.  The caller provides a Pandas DataFrame,
        and an column name for x values.  The y values come from all remaining columns.
    
    :param df_in:    Pandas DataFrame that has an x and multiple y columns.
    :param x_column:    DataFrame column that holds x axis values
    :param plot_title:    Title of Graph.
    :param y_left_label:    left y axis label
    :param y_right_label:    right y axis label
    :param bar_plot:    If True, create a Bar plot figure.  Otherwise, create a Scatter figure.
    :param figsize:    like the matplotlib figsize parameter
    :param number_of_ticks_display:    The number of x axis ticks to display. (Default = 20)
    :param yaxis2_cols:    A list of columns that contain y values that will be graphed 
                            versus the left y axis.
    '''
    ya2c = [] if yaxis2_cols is None else yaxis2_cols
    ycols = [c for c in df_in.columns.values if c != x_column]
    # create tdvals, which will have x axis labels
    td = list(df_in[x_column]) 
    nt = len(df_in)-1 if number_of_ticks_display > len(df_in) else number_of_ticks_display
    spacing = len(td)//nt
    tdvals = td[::spacing]
    
    # create data for graph
    data = []
    # iterate through all ycols to append to data that gets passed to go.Figure
    for ycol in ycols:
        if bar_plot:
            b = go.Bar(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
            if marker_color is not None:                
                b.marker=dict(color=marker_color)                
        else:
            b = go.Scatter(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        data.append(b)

    # create a layout
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(
            ticktext=tdvals,
            tickvals=tdvals,
            tickangle=45,
            type='category'),
        yaxis=dict(
            title='y main' if y_left_label is None else y_left_label
        ),
        yaxis2=dict(
            title='y alt' if y_right_label is None else y_right_label,
            overlaying='y',
            side='right'),
        margin=Margin(
            b=100
        )        
    )

    fig = go.Figure(data=data,layout=layout)
    return fig

def make_chart_html(figure_id,df,x_column,**kwdargs):
    '''
    Create a dash_html_components.Div wrapper of the dash_core_components.Graph instance.
    
    :param figure_id:    The id parameter of the plotly.graph_objs.Figure instance, that
                            get's created by plotly_plot, that you will display.
    :param df:    DataFrame that will act as input to a call of plotly_plot to create a
                    plotly.graph_objs.Graph figure.
    :param x_column:    Column that holds x values, used in call to plotly_plot.
    '''
    f = go.Figure()
    if df is not None and len(df)>0:
        f = plotly_plot(df,x_column,**kwdargs)
    gr = dcc.Graph(
            id=figure_id,
            figure=f,               
            )
    gr_html = html.Div(
        gr,
        className='item1',
        style={'margin-right':'auto' ,'margin-left':'auto' ,
               'height': '98%','width':'98%','border':'thin solid'},
        id = f'{figure_id}_html'
    )
    return gr_html


class PlotlyCandles():
    BAR_WIDTH=.5
    def __init__(self,df,title='candle plot',number_of_ticks_display=20):
        self.df = df.copy()
        #  and make sure the first index is 1 NOT 0!!!!!!
        self.df.index = np.array(list(range(len(df))))+1
        
        self.title = title
        self.number_of_ticks_display = number_of_ticks_display
        
    def get_candle_shapes(self):
        df = self.df.copy()
        xvals = df.index.values #chg    
        lows = df.low.values
        highs = df.high.values
        closes = df.close.values
        opens = df.open.values
        df['is_red'] = df.open>=df.close
        is_reds = df.is_red.values
        lines_below_box = [{
                    'type': 'line',
                    'x0': xvals[i],
                    'y0': lows[i],
                    'x1': xvals[i],
                    'y1': closes[i] if is_reds[i] else opens[i],
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 1.5,
                    }
                } for i in range(len(xvals))
        ]

        lines_above_box = [{
                    'type': 'line',
                    'x0': xvals[i],
                    'y0': opens[i] if is_reds[i] else closes[i],
                    'x1': xvals[i],
                    'y1': highs[i],
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 1.5,
                    }
                }for i in range(len(xvals))
        ]


        boxes = [{
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': xvals[i]- PlotlyCandles.BAR_WIDTH/2,
                    'y0': closes[i] if is_reds[i] else opens[i],
                    'x1': xvals[i]+ PlotlyCandles.BAR_WIDTH/2,
                    'y1': opens[i] if is_reds[i] else closes[i],
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 1,
                    },
                    'fillcolor': 'rgba(255, 0, 0, 0.6)' if is_reds[i] else 'rgba(0, 204, 0, 0.6)',
                } for i in range(len(xvals))
        ]
        shapes = lines_below_box + boxes + lines_above_box
        return shapes

    
    def get_figure(self):
        '''
        Use Plotly to create a financial candlestick chart.
        The DataFrame df_in must have columns called:
         'date','open','high','low','close'
        '''
        # Step 0: get important constructor values (so you don't type 'self.'' too many times)
        df_in = self.df.copy()
        title=self.title
        number_of_ticks_display=self.number_of_ticks_display
        
        # Step 1: only get the relevant columns and sort by date
        cols_to_keep = ['date','open','high','low','close','volume']
        df = df_in[cols_to_keep].sort_values('date')
        # Step 2: create a data frame for "green body" days and "red body" days
        # Step 3: create the candle shapes that surround the scatter plot in trace1
        shapes = self.get_candle_shapes()

        # Step 4: create an array of x values that you want to show on the xaxis
        spaces = len(df)//number_of_ticks_display
        indices = list(df.index.values[::spaces]) + [max(df.index.values)]
        tdvals = df.loc[indices].date.values

        # Step 5: create a layout
        layout1 = go.Layout(
            showlegend=False,
            title = title,
            margin = dict(t=100),
            xaxis = go.layout.XAxis(
                tickmode = 'array',
                tickvals = indices,
                ticktext = tdvals,
                tickangle=90,
                showgrid = True,
                showticklabels=True,
                anchor='y2', 
            ),       
            yaxis1 = go.layout.YAxis(
                range =  [min(df.low.values), max(df.high.values)],
                domain=[.22,1]
            ),
            yaxis2 = go.layout.YAxis(
                range =  [0, max(df.volume.values)],
                domain=[0,.2]
            ),
            shapes = shapes
        )

        # Step 6: create a scatter object, and put it into an array
        def __hover_text(r):
            d = r.date
            o = r.open
            h = r.high
            l = r.low
            c = r.close
            v = r.volume
            t = f'date: {d}<br>open: {o}<br>high: {h}<br>low: {l}<br>close: {c}<br>volume: {v}' 
            return t
        df['hover_text'] = df.apply(__hover_text,axis=1)
        hover_text = df.hover_text.values

        # Step 7: create scatter (close values) trace.  The candle shapes will surround the scatter trace
        trace1 = go.Scatter(
            x=df.index.values,
            y=df.close.values,
            mode = 'markers',
            text = hover_text,
            hoverinfo = 'text',
            xaxis='x',
            yaxis='y1'
        )

        # Step 8: create the bar trace (volume values)
        trace2 = go.Bar(
            x=df.index.values,
            y=df.volume.values,
            width = PlotlyCandles.BAR_WIDTH,
            xaxis='x',
            yaxis='y2'
        )

        # Step 9: create the final figure and pass it back to the caller
        fig1 = {'data':[trace1,trace2],'layout':layout1}
        return fig1
    
    def plot(self):
        fig = self.get_figure()
        iplot(fig)
        return fig


def flatten_layout(app):
    # define recursive search
    def _recursive_div_display(c,div_list):
    #     pdb.set_trace()
        if  hasattr(c,'children') and 'div.div' in str(type(c)).lower() and len(c.children)>0:
            for c2 in c.children:
                _recursive_div_display(c2,div_list)
        else:
            div_list.append(c)
    
    # run recursive search
    final_list = []
    for c in list(np.array(app.layout.children).reshape(-1)):    
        dlist = []
        _recursive_div_display(c,dlist)
        final_list.extend(dlist)
    
    # return results
    return final_list
    
# ***************************************** Define main dash core and html component wrapper ***********************************
class  ComponentWrapper():
    @staticmethod
    def build_from_json(component_json):
        cbt = None if 'callback_input_transformer' not in component_json else component_json['callback_input_transformer']
        ict = None if 'input_component_tuples' not in component_json else component_json['input_component_tuples']
        cw = ComponentWrapper(component_json['component'], 
                component_json['properties_to_output'], 
                input_component_tuples=ict, 
                callback_input_transformer=cbt, 
                style=None if 'style' not in component_json else component_json['style'])
        return cw
        
    
    def __init__(self,
                 dash_component,
                 input__tuples=None,
                 output_tuples=None,
                 callback_input_transformer=None,
                 style=None,
                 logger=None,
                 loading_state='cube'):
        self.logger = init_root_logger(DEFAULT_LOG_PATH, DEFAULT_LOG_LEVEL) if logger is None else logger
        self.component = dash_component
        self.cid = self.component.id
        self.id = self.cid
        self.html_id = f'{self.cid}_html'
        self.properties_to_output = [] if output_tuples is None else output_tuples
        # create callback output list
        self.output_tuples = output_tuples   
        self.output_data_tuple = None 
        if output_tuples is not None:
            for ot in output_tuples:
                if ot[1] == 'data' and self.output_data_tuple is None:
                    self.output_data_tuple = ot
                      
        self.callback_outputs = []
        for p in self.properties_to_output:
            if type(p)==tuple:
                t = p
#                 o = Output(*p)
            else:
#                 o = Output(self.cid,p)
                t = (self.id,p)
            o = Output(*t)
            self.callback_outputs.append(o)
        
        # create callback input list  
        self.callback_inputs = []
        if input__tuples is not None:
            for ict in input__tuples:
                ic_id = ict[0]
                p = ict[1]
                self.callback_inputs += [Input(ic_id,p)]
                
        self.style = {} if style is None else style
        
        self.callback_input_transformer = callback_input_transformer
        def _default_transform(callback_input_list):
            return callback_input_list
        
        if self.callback_input_transformer is None:
            self.callback_input_transformer = _default_transform
        self.div = html.Div([self.component])
#         self.div = dcc.Loading(children=html.Div([self.component]),type=loading_state)


        
    @property
    def html(self):
        return self.div           

    def callback(self,theapp):     
        @theapp.callback(
            self.callback_outputs, 
            self.callback_inputs 
            )
        def execute_callback(*inputs_and_states):
            l = list(inputs_and_states)
            self.logger.debug(f'{self.html_id} input: {l}')
            ret = self.callback_input_transformer(l)
            self.logger.debug(f'{self.html_id} output: {ret}')
            return ret
        if len(self.callback_inputs)<=0:
            return None     
        return execute_callback

def stop_callback(errmess,logger=None):
    m = "****************************** " + errmess + " ***************************************"     
    if logger is not None:
        logger.debug(m)
    raise PreventUpdate()
#     raise ValueError(m)
# ************************ Define the classes that inherit dgrid.ComponentWrapper ************************

class DivComponent(ComponentWrapper):
    def __init__(self,component_id,input_component=None,
                 input_component_property='data',
                 initial_children=None,
                 callback_input_transformer=None,
                 style=None,logger=None):
        s = border_style if style is None else style
        init_children = '' if initial_children is None else initial_children
        h1 = html.Div(init_children,id=component_id,style=s)
        h1_lambda = (lambda v:[v]) if callback_input_transformer is None else callback_input_transformer
        input_tuple = None if input_component is None else [(input_component.id,input_component_property)]
        super(DivComponent,self).__init__(
                    h1,input__tuples=input_tuple,
                    output_tuples=[(h1.id,'children')],
                    callback_input_transformer=h1_lambda,
                    style=s,
                    logger=logger)

    
markdown_style={
    'borderWidth': '1px',
    'borderStyle': 'solid',
    'borderRadius': '1px',
    'background-color':'#ffffff',
}
class MarkdownComponent(ComponentWrapper):
    def __init__(self,component_id,
                 markdown_text,
                 style=None,logger=None):
        super(MarkdownComponent,self).__init__(
            html.Span(dcc.Markdown(markdown_text),id=component_id,
            style=markdown_style),logger=logger)

class UploadComponent(ComponentWrapper):
    def __init__(self,component_id,text=None,
                 initial_data = None,
                 acceptable_file_extensions='.csv',
                 style=None,logger=None):
        t = "Choose a File" if text is None else text
        self.component_id = component_id
        u1 = dcc.Upload(
                    id=component_id,
                    children=html.Div([t]),
                    accept = acceptable_file_extensions,
                    # Allow multiple files to be uploaded
                    multiple=False,
                    style=blue_button_style if style is None else style)
        
        u1_lambda = lambda value_list: [None] if value_list[0] is None else [file_store_transformer(value_list[0])]
        self.s1 = dcc.Store(id=component_id+"_store",data=[] if initial_data is None else [initial_data])
        super(UploadComponent,self).__init__(
                    u1,input__tuples=[(u1.id,'contents')],
                    output_tuples=[(self.s1.id,'data')],
                    callback_input_transformer=u1_lambda,logger=logger)
        
    @ComponentWrapper.html.getter
    def html(self):
        return html.Div([self.div,self.s1])

class UploadFileNameDiv(DivComponent):
    def __init__(self,component_id,upload_component,text=None,style=None,initial_file_to_show=None):
        t = 'YOU ARE VIEWING:' if text is None else text
        initial_text = '' if initial_file_to_show is None else initial_file_to_show
        super(UploadFileNameDiv,self).__init__(component_id,upload_component,
                input_component_property='filename',
                initial_children=initial_text,
                style=blue_button_style if style is None else style,
                callback_input_transformer=lambda v: [f'{t} {v}'])


    
class DashTableComponent(ComponentWrapper):
    def __init__(self,component_id,df_initial,input_component=None,title=None,
                 transform_input=None,editable_columns=None,style=None,logger=None,
                 columns_to_round=None,digits_to_round=2,title_style=None):
        
        self.logger = init_root_logger(DEFAULT_LOG_PATH, DEFAULT_LOG_LEVEL) if logger is None else logger
        # add component_id and html_id to self
        self.component_id = component_id
        self.html_id = self.component_id+'_html'
        self.columns_to_round = columns_to_round
        self.digits_to_round = digits_to_round
        self.title_style = title_style
        
        
        default_data_store = []
        if input_component is None:
            dcs_id = f'{component_id}_default_store'
#             dcs_data = None if df_initial is None else df_initial.to_dict('rows')
            dcs_data = None if df_initial is None else df_initial.to_dict()
            default_data_store.append(dcc.Store(id=dcs_id,data=dcs_data))
            input_tuples = [(dcs_id,'data')]
        else:
            input_tuples = [input_component.output_data_tuple]
        
        # create initial div
        cols = None if df_initial is None else df_initial.columns.values
        dtable_div = create_dt_div(component_id,df_in=df_initial,
                        columns_to_display=cols,
                        editable_columns_in=editable_columns,
                        title='Dash Table' if title is None else title,
                        title_style=self.title_style)
        
        dt_children = [dtable_div] + default_data_store
        outer_div = html.Div(dt_children,id=self.html_id,
                style={
                'margin-right':'auto' ,'margin-left':'auto' ,
                'height': '98%','width':'98%','border':'thin solid'})
        s = border_style if style is None else style
        for k in s:
            outer_div.style[k] = s[k]
        
        # define dash_table callback using closure so that you don't refer back to 
        #   class instance during the callback
        def _create_dt_lambda(component_id,cols,editable_cols,logger,transform_input=None,title=None):
            def _dt_lambda(value_list): 
                logger.debug(f'{component_id} entering _dt_lambda value_list: {value_list}')
                ret = [None,None]
                try:
                    if value_list[0] is not None:
                        input_dict = value_list[0]
                        if transform_input is not None:
                            df = transform_input(input_dict)
                        else:
                            df = make_df(input_dict)
                        # check if rounding is  necessary
                        if self.columns_to_round is not None:
                            df = dataframe_rounder(df, digits=self.digits_to_round, columns_to_round=self.columns_to_round)
                        dt_div = create_dt_div(component_id,df_in=df,
                                columns_to_display=cols,
                                editable_columns_in=editable_cols,
                                title=title,
                                logger=logger,
                                title_style=self.title_style)
                        if df is None or len(df)<1:
                            ret = [None,None]
                        else:
                            output_dict = df.to_dict('records')
                            ret =  [dt_div,output_dict]
                except BadColumnsException:
                    # return an error message in place of the actual DataFrame
                    colstext = str(cols).replace('[','').replace(']','')
                    children = [
                        html.H3('ERROR FROM INPUT CSV FILE'),
                        html.P(f'Expected CSV File with columns:'),
                        html.P(colstext),
                        html.P(f'Actual CSV Columns are:'),
                        html.P(str(df.columns.values))
                    ]
                    err_html = html.Div(children,style={'line-height':'100%'})
                    ret = [err_html,{}]                    
                except Exception as e:
                    logger.warn(f'!!!!!!!!!!!!!! {component_id} _dt_lambda EXCEPTION: {str(e)} !!!!!!!!!!!!!')
                    traceback.print_exc()
                logger.debug(f'{component_id} _dt_lambda ret: {ret}')
                if ret[0] is None:
                    stop_callback(f'{component_id} _dt_lambda value_list has no data.  Callback return is ignored',logger)
                return ret
            return _dt_lambda
        
        
        # do super, but WITHOUT the callback
        super(DashTableComponent,self).__init__(outer_div,
                     input__tuples=input_tuples,
                     output_tuples=[
                         (outer_div.id,'children'),
                         (self.component_id,'data')],
                     callback_input_transformer=lambda v:[None],logger=logger)

        # set the id of this class instance because the input component 
        #   to the super constructor is NOT the main dcc component
        self.id = component_id
        
        # define callback so that it includes self.logger
        dtlam = _create_dt_lambda(self.component_id,cols,
                                  editable_columns,self.logger,transform_input=transform_input,title=title)
        self.callback_input_transformer = dtlam
    
    
            
class XyGraphComponent(ComponentWrapper):
    def __init__(self,component_id,input_component,x_column,
                 transform_input=None,
                 plot_bars=True,title=None,
                 marker_color=None,
                 style=None,logger=None):
        
        self.logger = init_root_logger(DEFAULT_LOG_PATH, DEFAULT_LOG_LEVEL) if logger is None else logger
        # create title
        t = f"Graph {component_id}" if title is None else title
        
        # get input tuple
        input_tuple = input_component.output_data_tuple

        # add component_id and html_id to self
        self.component_id = component_id
        gr_html = make_chart_html(component_id,None,
                    x_column,plot_title=t,marker_color=marker_color)

        # define dash_table callback using closure so that you don't refer back to 
        #   class instance during the callback
        def _create_gr_lambda(component_id,x_column,plot_title,logger,transform_input=None):
            def gr_lambda(value_list): 
                logger.debug(f'{component_id} gr_lambda value_list: {value_list}')
                ret = [None]
                try:
                    if value_list is not None and len(value_list)>0 and value_list[0] is not None:
                        if transform_input is not None:
                            df = transform_input(value_list[0])
                        else:
                            df = make_df(value_list[0])
                        if df is not None:
                            fig = plotly_plot(df,x_column,
                                    plot_title=plot_title,bar_plot=plot_bars,marker_color=marker_color)
                            ret =  [fig]
                except Exception as e:
                    traceback.print_exc()
                    logger.warn(f'gr_lambda ERROR:  {str(e)}')
                logger.debug(f'{component_id} gr_lambda ret: {ret}')
                if ret[0] is None:
                    err_mess = f'{component_id} gr_lambda IGNORING callback. NO ERROR'
                    stop_callback(err_mess,logger)
                return ret
            return gr_lambda

        # set the outer html id
        gr_html.style = border_style if style is None else style
        # do super, but WITHOUT the callback
        super(XyGraphComponent,self).__init__(gr_html,
                     input__tuples=[input_tuple],
                     output_tuples=[(self.component_id,'figure')],
                     callback_input_transformer=lambda v:[None])
        
        # define callback so that it includes self.logger
        gr_lam = _create_gr_lambda(self.component_id,x_column,t,self.logger,transform_input=transform_input)
        self.callback_input_transformer = gr_lam
        
#**************************************************************************************************
class ChainedDropDownDiv(ComponentWrapper):
    def __init__(self,component_id,
                 dropdown_input_components=None,
                 initial_dropdown_labels=None,
                 initial_dropdown_values=None,
                 placeholder = None,
                 choices_transformer_method=None,
                 default_initial_index=0,
                 style=None,logger=None):
        
        self.logger = init_root_logger(DEFAULT_LOG_PATH, DEFAULT_LOG_LEVEL) if logger is None else logger

        # add component_id to self
        self.component_id = component_id
        self.dropdown_id = f'{component_id}_dropdown'
        self.style = button_style if style is None else style
        
        # define callback inputs
        # let the dropdown choice that one makes be the first input to the list of callback inputs
        input_tuples = [(self.dropdown_id,'value')]
        # if there is another callback input, add it to the list of callback inputs
        if dropdown_input_components is not None:
            for dic in dropdown_input_components:
                input_tuples += [dic.output_data_tuple]

        # if there are predefined labels and values for the dropdown, assign them                
        self.dropdown_choices = [] if initial_dropdown_labels is None else [{'label':l,'value':v} for l,v in zip(initial_dropdown_labels,initial_dropdown_values)]
        if len(self.dropdown_choices)>0:
            # create dropdown dash component with initial choices
            self.dropdown = dcc.Dropdown(id=self.dropdown_id, value=initial_dropdown_values[default_initial_index],
                    options=self.dropdown_choices,
                    placeholder="Select an Option" if placeholder is None else placeholder,
                    style=self.style)
        else:
            # create dropdown dash component WITHOUT initial choices (they will be added during the callback)
            self.dropdown = dcc.Dropdown(id=self.dropdown_id,
                    placeholder="Select an Option" if placeholder is None else placeholder,
                    style=self.style)
            
        self.dropdown_div = html.Div([self.dropdown])
        self.input_transformer_method = lambda v: v[-1]
        self.dcc_id = f'{component_id}_dropdown_output'
        self.dcc_store = dcc.Store(id=self.dcc_id)
        output_tuples = [(self.dcc_id,'data'),(self.dropdown.id,'options')]
        
        # assign or create the method that dynamically creates dropdown choices
        self.choices_transformer_method = choices_transformer_method
        if choices_transformer_method is None:
            self.choices_transformer_method = lambda _: self.dropdown_choices

        
        self.fd_div = html.Div([self.dropdown_div,self.dcc_store])
        self.current_value = None
        def _create_transformer_lambda(choices_transformer_method,component_id,logger):            
            def _dropdown_transformer(v):
                logger.debug(f'_dropdown_transformer {component_id} input: {v}') 
                new_choices =  choices_transformer_method(v)
                selected_item =  v[0]               
                return [selected_item,new_choices]
            return _dropdown_transformer

        super(ChainedDropDownDiv,self).__init__(self.dropdown,
                     input__tuples=input_tuples,
                     output_tuples=output_tuples,
                     callback_input_transformer=lambda v:[None],logger=logger)
        
        # use the _create_transformer_lambda to inject values into the method that the callback will use
        dd_lam = _create_transformer_lambda(self.choices_transformer_method,self.component_id,self.logger)
        # set the callback method (after the parent ComponentWrapper has been instantiated
        self.callback_input_transformer = dd_lam

    @ComponentWrapper.html.getter
    def html(self):
        return self.fd_div

#**************************************************************************************************



class FigureComponent(ComponentWrapper):
    def __init__(self,component_id,
                 input_component,
                 create_figure_from_df_transformer,
                 input_component_property='data',
                 figure=None,style=None,logger=None):

        self.logger = init_root_logger(DEFAULT_LOG_PATH, DEFAULT_LOG_LEVEL) if logger is None else logger

        # add component_id and html_id to self
        self.component_id = component_id
        gr_html = make_chart_html(component_id,None,None)
        self.figure = figure
        
        def _create_gr_lambda(component_id,_figure_from_df_transformer,logger,hard_coded_figure=None):
            def gr_lambda(value_list): 
                logger.debug(f'{component_id} gr_lambda value_list: {value_list}')
                ret = [None]
                try:
                    if value_list is not None and len(value_list)>0 or value_list[0] is not None:
                        ret =  [_figure_from_df_transformer(value_list)]
                except Exception as e:
                    traceback.print_exc()
                    logger.warn(f'gr_lambda ERROR:  {str(e)}')
                logger.debug(f'{component_id} gr_lambda ret: {ret}')
                if ret[0] is None:
                    err_mess = f'{component_id} gr_lambda IGNORING callback. NO ERROR'
                    stop_callback(err_mess,logger)
                return ret
            
            if hard_coded_figure is not None:
                return lambda _:[hard_coded_figure]    
            return gr_lambda

        # set the outer html id
        gr_html.style = border_style if style is None else style
        
        # do super, but WITHOUT the callback
        super(FigureComponent,self).__init__(gr_html,
                     input__tuples=[(input_component.id,input_component_property)],
                     output_tuples=[(self.component_id,'figure')],
                     callback_input_transformer=lambda v:[None])
        
        # define callback so that it includes self.logger
        gr_lam = _create_gr_lambda(self.component_id,create_figure_from_df_transformer,
                                   self.logger)
        self.callback_input_transformer = gr_lam



class StoreComponent(ComponentWrapper):        
    def __init__(self,component_id,
                 input_component,
                 create_data_dictionary_from_df_transformer,
                 input_component_property='data',
                 initial_data = None,
                 logger=None):
        
        if type(input_component)==type([]):
            # in this case, input_component is a list of tuples
            input_tuples = input_component
        else:
            input_tuples = [(input_component.id,input_component_property)]
        
        dcc_store = dcc.Store(id=component_id,data=[] if initial_data is None else [initial_data])
        
        def _create_callback_lambda(component_id,_input_transformer,logger,**kwargs):
            def callback_lambda(value_list): 
                s = f'{component_id} entering callback value_list: {value_list}'
                logger.debug(s)
                ret = [None]
                try:
                    if value_list is not None and len(value_list)>0 or value_list[0] is not None:
                        ret =  [_input_transformer(value_list,**kwargs)]
                except Exception as e:
                    traceback.print_exc()
                    logger.warn(f'{component_id}  _create_store_data_lambda ERROR:  {str(e)}')
                logger.debug(f'{component_id} callback ret: {ret}')
                if ret[0] is None:
                    err_mess = f'{component_id} callback: IGNORING callback. NO ERROR'
                    stop_callback(err_mess,logger)
                return ret
            return callback_lambda
        
        # do super
        super(StoreComponent,self).__init__(dcc_store,
#                      input__tuples=[(input_component.id,input_component_property)],
                     input__tuples=input_tuples,
                     output_tuples=[(component_id,'data')],
                     callback_input_transformer = lambda v: [None],
                     style={'display':'none'},
                     logger=logger)
        
        self.callback_input_transformer  = _create_callback_lambda(component_id,
                                create_data_dictionary_from_df_transformer, self.logger) 
        self.dcc_store = dcc_store
        
    @ComponentWrapper.html.getter
    def html(self):
        return dcc.Loading(children=[self.dcc_store],type='cube')

class FiledownloadComponent(ComponentWrapper):
    def __init__(self,component_id,
                 dropdown_labels,dropdown_values,
                 drop_down_placeholder_text,a_link_text,
                 create_file_name_transformer=None,
                 style=None,logger=None):

        # create important id's
        self.dropdown_id = component_id+"_dropdown"
        self.a_link_id = component_id + '_last_downloaded'

        s = button_style if style is None else style
        self.dropdown_values = dropdown_values
        dropdown_choices = [{'label':l,'value':v} for l,v in zip(dropdown_labels,dropdown_values)]
        dropdown_div = html.Div([
                dcc.Dropdown(
                    id=self.dropdown_id, 
                    options=dropdown_choices,
                    style=s,
                    placeholder=drop_down_placeholder_text)
        ])
        
        # creat A div 
        href_div = html.Div(html.A(a_link_text,href='',id=self.a_link_id),style=s)
        gs= grid_style.copy()
        gs['background-color'] = '#ffffff'
        self.fd_div = html.Div([dropdown_div,href_div],style=gs,id=component_id)
        self.create_file_name_transformer = lambda value: str(value) if create_file_name_transformer is None else create_file_name_transformer
        
        # create callback that populates the A link
        def _update_link(input_value):
            v = input_value[0]
            if v is None:
                v = self.dropdown_values[0]
            return ['/dash/urlToDownload?value={}'.format(v)]        
                
        super(FiledownloadComponent,self).__init__(
            self.fd_div, 
            input__tuples=[(self.dropdown_id,'value')], 
            output_tuples=[(self.a_link_id,'href')], 
            callback_input_transformer=_update_link, 
            logger=logger)
    
    def route(self,theapp):
        @theapp.server.route('/dash/urlToDownload')
        def download_csv():
            value = flask.request.args.get('value')            
            fn = self.create_file_name_transformer(value)
            print(f'FileDownLoadDiv callback file name = {fn}')
            return flask.send_file(fn,
                               mimetype='text/csv',
                               attachment_filename=fn,
                               as_attachment=True)
        return download_csv


class FileDownLoadDiv():
    def __init__(self,html_id,
                 dropdown_labels,dropdown_values,drop_down_placeholder_text,a_link_text,
                 create_file_name_transformer=None,
                 style=None):
        self.html_id = html_id
        s = button_style if style is None else style
        self.input_tuple = (f'{html_id}_dropdown','value')
        self.dropdown_values = dropdown_values
        dropdown_choices = [{'label':l,'value':v} for l,v in zip(dropdown_labels,dropdown_values)]
        dropdown_div = html.Div([
                dcc.Dropdown(
                    id=self.input_tuple[0], 
#                     value=dropdown_values[0],
                    options=dropdown_choices,
                    style=s,
                    placeholder=drop_down_placeholder_text)
        ])
        self.output_tuple = (f'{html_id}_last_downloaded','href')
        href_div = html.Div(html.A(a_link_text,href='',id=self.output_tuple[0]),style=s)
        gs= grid_style.copy()
        gs['background-color'] = '#ffffff'
        self.fd_div = html.Div([dropdown_div,href_div],style=gs)
        self.create_file_name_transformer = lambda value: str(value) if create_file_name_transformer is None else create_file_name_transformer
    @property
    def html(self):
        return self.fd_div
        

    def callback(self,theapp):     
        @theapp.callback(
            Output(self.output_tuple[0], self.output_tuple[1]), 
            [Input(self.input_tuple[0],self.input_tuple[1])]
            )
        def update_link(value):
            v = value
            if v is None:
                v = self.dropdown_values[0]
            return '/dash/urlToDownload?value={}'.format(v)        
        return update_link
    
    def route(self,theapp):
        @theapp.server.route('/dash/urlToDownload')
        def download_csv():
            value = flask.request.args.get('value')            
            fn = self.create_file_name_transformer(value)
            print(f'FileDownLoadDiv callback file name = {fn}')
            return flask.send_file(fn,
                               mimetype='text/csv',
                               attachment_filename=fn,
                               as_attachment=True)
        return download_csv


def recursive_grid_layout(app_component_list,current_component_index,gtcl,layout_components,wrap_in_loading_state=False):
    # loop through the gtcl, and assign components to grids
    for grid_template_columns in gtcl:
        if type(grid_template_columns)==list:
            layout_components.append(recursive_grid_layout(
                app_component_list,current_component_index, grid_template_columns, 
#                 layout_components,wrap_in_loading_state=True))
                layout_components,wrap_in_loading_state=False))
            continue
        # if this grid_template_columns item is NOT a list, process it normally
        sub_list_grid_components = []
        num_of_components_in_sublist = len(grid_template_columns.split(' '))
        for _ in range(num_of_components_in_sublist):
            # get the current component
            layout_ac = app_component_list[current_component_index]
            # add either the component, or it's html property to the sublist
            if hasattr(layout_ac, 'html'):
                layout_ac = layout_ac.html
            sub_list_grid_components.append(layout_ac)
            current_component_index +=1
        new_grid = create_grid(sub_list_grid_components, 
                        additional_grid_properties_dict={'grid-template-columns':grid_template_columns},
                        wrap_in_loading_state=wrap_in_loading_state)
        layout_components.append(new_grid)



def make_app(app_component_list,grid_template_columns_list=None,app=None):
    components_with_callbacks = []
    layout_components = []
    
    # get layout template list (lot)
    default_gtcl = ('1fr '* len(app_component_list))[:-1]
    gtcl = default_gtcl if grid_template_columns_list is None else grid_template_columns_list
    
    # populate layout_components using recursive algo
    recursive_grid_layout(app_component_list,0,gtcl,layout_components)
    ret_app =   dash.Dash() if app is None else app
    ret_app.layout = html.Div(layout_components,style={'margin-left':'10px','margin-right':'10px'})
    
    ret_app =   dash.Dash() if app is None else app
    ret_app.layout = html.Div(layout_components,style={'margin-left':'10px','margin-right':'10px'})

    for ac in app_component_list:
        if hasattr(ac, 'route'):
            ac.route(ret_app)
    
    for ac in app_component_list:
        if isinstance(ac, ComponentWrapper):
            components_with_callbacks.append(ac)
    [c.callback(ret_app) for c in components_with_callbacks]
    return ret_app
        
def make_multi_page_app(page_dict,app=None):
    layout_dict = {}
    ret_app =   dash.Dash() if app is None else app
    ret_app.config.suppress_callback_exceptions = True

    for page_address in page_dict.keys():
        page_sub_dict = page_dict[page_address]
        grid_template_columns_list = page_sub_dict['gtcl']
        app_component_list = page_sub_dict['app_component_list']
        make_app(app_component_list,
                            grid_template_columns_list,ret_app)
        layout_dict[page_address] = ret_app.layout
        ret_app.layout=html.Div()
     
    ret_app.layout = html.Div([
        dcc.Location(id='url',refresh=False),
        html.Div(id='page_content')])
    
    @ret_app.callback(
        Output('page_content', 'children'),
        [Input('url', 'pathname')])
    def display_page(pathname):
#         return 'hello world'
        ret_layout = layout_dict[pathname]
        return ret_layout
    
    return ret_app
    

