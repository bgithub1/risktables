import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io

import pandas as pd
import datetime
import pytz
import var_models
import risk_tables
from IPython import display


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
	rt = risk_tables.RiskCalcs(use_postgres=False)
	var_results = rt.calculate(df_port)
	return_dict = {}
	for k in var_results.keys():
		r = var_results[k]
		if k[:2] == 'df':
			 r = pd.DataFrame(r).to_dict(orient="records")
		return_dict[k] = r
	return return_dict

