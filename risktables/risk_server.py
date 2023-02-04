import pandas as pd
from fastapi import Depends, FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from pydantic import BaseModel

from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn

import pandas as pd
import datetime
import pytz
import var_models
import risk_tables
from IPython import display
import argparse

# use this for global variables that are referenced in the main
class GlobalVariables:
    pass

__m = GlobalVariables()  # m will contain all module-level values
__m.redis_port = None  # database name global in module



app = FastAPI()

# code goes here
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://www.billybyte.com",
    "https://www.billybyte.com",
    "http://billybyte.com",
    "https://billybyte.com",
    "http://localhost",
    "http://localhost:3010",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Risk Tables Server"}

@app.get("/get_var")
async def get_var():
	df_port = pd.read_csv('spdr_stocks.csv')
	var_model = var_models.VarModel(df_port)
	var_results = var_model.compute_var()
	return_dict = {}
	for k in var_results.keys():
		r = var_results[k]
		if type(r) == pd.DataFrame:
			r = r.to_dict(orient="records")
		return_dict[k] = r
	return return_dict


@app.get("/get_risk")
async def get_risk_tables():
	df_port = pd.read_csv('spdr_stocks.csv')
	rt = risk_tables.RiskCalcs(use_postgres=False,redis_port = __m.redis_port)
	var_results = rt.calculate(df_port)
	return_dict = {}
	for k in var_results.keys():
		r = var_results[k]
		if k[:2] == 'df':
			 r = pd.DataFrame(r).to_dict(orient="records")
		return_dict[k] = r
	return return_dict

class CsvData(BaseModel):
    data: str

@app.post("/upload_csv")
async def get_risk_tables_from_csv(csv_data: CsvData):
	csv_text = csv_data.data
	print(type(csv_text))
	print(csv_text)
	list_data = csv_text.split(';')
	list_data = [
		v.split(',')
		for v in list_data
		if len(v)>0
	]
	dict_data = [
		{'symbol':v[0],'position':int(v[1])}
		for v in list_data[1:]
	]
	df_port = pd.DataFrame(dict_data)
	rt = risk_tables.RiskCalcs(use_postgres=False,redis_port=__m.redis_port)
	var_results = rt.calculate(df_port)
	return_dict = {}
	for k in var_results.keys():
		r = var_results[k]
		if k[:2] == 'df':
			 r = pd.DataFrame(r).to_dict(orient="records")
		return_dict[k] = r
	return return_dict


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			prog = 'risk_server',
			description = 'A FastAPI restAPI to server pricing and risk information from a portfolio',
			)
	hour = datetime.datetime.now().hour
	parser.add_argument('--host',default='127.0.0.1',type=str,help="uvicorn http host") 
	parser.add_argument('--port',default=8555,type=int,help="uvicorn http port") 
	parser.add_argument('--reload',
		help="Tell uvicorn to automatically reload server if source code changes",
		action='store_true'
	) 
	parser.add_argument('--log_level',default='info',type=str,
			help="the logger's log level")
	parser.add_argument('--redis_port',default=None,type=int,
		help="Redis port, if you are going to use Redis instead of Yahoo to fetch data") 
	args = parser.parse_args()  
	print(args)
	__m.redis_port = args.redis_port

	uvicorn.run(
		"risk_server:app", 
		host=args.host, 
		port=args.port, 
		reload=args.reload, 
		log_level=args.log_level
	)

