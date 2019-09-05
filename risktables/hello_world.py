'''
Created on Sep 5, 2019

@author: bperlman1
'''
import dash
import dash_html_components as html
import dash_core_components as dcc
import sys

if __name__=='__main__':
    host = sys.argv[1]
    port = sys.argv[2]
    app = dash.Dash()
    app.layout = html.Div('hello world')
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True    
    app.run_server(host=host,port=port)
