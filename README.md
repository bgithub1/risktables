### risktables: create portfolio risk measures, and present them using Plot.ly Dash
REQUIRES Python 3.6 or above
___
The risktables python project contains modules to generate portfolio risk measures like VaR, Greeks, Correlation matrices, etc..  

As well, it contains a Plot.ly Dash app in the module ```dash_risk_v01.py```, which allows users to see those risk statistics on their own uploaded csv portfolio files.
___
### Quickstart
#### Installing the project:
1. git clone https://github.com/bgithub1/risktables.git
2. cd ./risktables
3. pip install -r requirements.txt

*You should now have a project folder called ```risktables```, and a package within that project also called ```risktables```*
___

### Run the dgrid_components_example_v01.ipynb jupyter notebook
1. Make sure you have Jupyter installed:
 * ```python3 -m pip install --upgrade pip```
 * ```python3 -m pip install jupyter```
2. cd risktables/risktables    # set your working directory to the risktables **PACKAGE**, as opposed to the risktables **project**
3. from a bash terminal, launch Jupyter
 * ```jupyter notebook```
4. When the jupyter notebook webpage appears, select the .ipynb file ```dgrid_components_example_v01.ipynb```
5. Run all the cells in this notebook.
 * The first several cells create a Pandas DataFrame called ```df_pseudo```, which creates open,high,low, and close data generated from a random walk, as well as a plotly python candlestick chart of that data.
 * The cells that follow the candlestick chart run a Plot.ly Dash web app that displays the same DataFrame and chart, using classes from the ```dgrid_components.py``` module.
6. Open a web browser and enter: ```localhost:8500``` in the address bar.
___

### Run the risk_example_1.ipynb notebook
1. Make sure you have Jupyter installed (see above)
2. cd risktables/risktables
3. from a bash terminal, launch Jupyter
 * ```jupyter notebook```
4. When the jupyter notebook webpage appears, select the .ipynb file ```risk_example_1.ipynb```
5. Run all the cells in this notebook.

___
### Launching the full portfolio risk web app
1. Navigate to risktables/risktables
2. python3.6  dash_risk_v01.py
3. Open a web browser and enter: ```localhost:8500``` in the address bar

#### About market data:
The web app uses yahoo finance data for daily history csv.  

If you have a free BarChart account, you can subscribe to the BarChart OnDemand API, which gives you API access to Commodities symbols, as well as index and currency pairs that are available on BarChart OnDemand. See https://www.barchart.com/ondemand/api for more info.

The web app also uses the Fred API for Federal Reserve Interest Rate Data.  You need to obtain an API key at https://research.stlouisfed.org/docs/api/api_key.html

#### Using BarChart Ondemand:
1. Create a BarChart Ondemand account
2. You will be given an API key.
3. Create a single line file called ```free_api_key.txt```, insert the API key, and place in risktables/risktables/temp_folder.

#### Using the FRED API:
1. In your web browser, navigate to https://research.stlouisfed.org/docs/api/api_key.html
2. Follow the instructions to create an API key
3. Create a single line file called ```fred_api_key.txt```, insert the API key, and place in risktables/risktables/temp_folder.
