'''
Created on Sep 5, 2019

@author: bperlman1
'''
import dash
import dash_html_components as html
import sys
import dash_auth

if __name__=='__main__':
    host = sys.argv[1]
    port = sys.argv[2]
    app = dash.Dash()
    VALID_USERNAME_PASSWORD_PAIRS = {
        'hello': 'world'
    }
    
    auth = dash_auth.BasicAuth(
        app,
        VALID_USERNAME_PASSWORD_PAIRS
    )    
    app.layout = html.Div('hello world')
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True    
    app.run_server(host=host,port=port)
