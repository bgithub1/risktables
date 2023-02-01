const express = require('express');
const bodyParser = require("body-parser");
const path = require('path');

// print process.argv
// process.argv.forEach((val, index) => {
//   console.log(`${index}: ${val}`);
// });

var riskdata_path = '/riskdata';
var riskdata_from_csv = '/riskdata_from_csv';

const pargs = process.argv;
if (pargs.length > 2) {
  riskdata_path =  pargs[2] + riskdata_path;
  riskdata_from_csv =  pargs[2] + riskdata_from_csv;
}



const fetch = (...args) =>
  import('node-fetch').then(({ default: fetch }) => fetch(...args));

const app = express();
const cols_conversions = {
  'symbol':'sym', 
  'position':'qty',
  'underlying':'usym',
  'close':'last',
  'stdev':'std',
  'delta':'del',
  'gamma':'gam',
  'unit_var':'uvar',
  'position_var':'pvar'
  };

// app.use(express.static('public'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

app.get(riskdata_path, (req, res) => {
  // const get_var_url = "http://localhost:8555/get_var";
  const get_var_url = "http://localhost:8555/get_risk";
  fetch(get_var_url)
    .then(response => response.json())
    .then(data => {
      // Do something with the data
      res.json(data);
    })
    .catch(error => {
      console.error('Error:', error);
      res.status(500).json({ error: 'An error occurred' });
    });
});


app.post(riskdata_from_csv, (req, res) => {
  // const get_var_url = "http://localhost:8555/get_var";
  r = '{"status":"okfromexpress"}';
	try {		
	  console.log(req.body.data);
	  console.log("success");
	} catch (error) {
	  console.error(Object.keys(req));
	  r = JSON.stringify({"status":error});
	}  
  const get_var_url = "http://localhost:8555/upload_csv";
  try {
	  fetch(get_var_url, {
	    method: 'POST',
	    headers: {
			'Accept': 'application/json',
			'Content-Type': 'application/json'
	    },
	    body: JSON.stringify({'data':req.body.data})
	    // body: {'data':req.body.data}
	  })
	    .then((response) => response.json())
	    .then((result) => {
	      console.log('Success:', result);
	      // res.json('{"status":"got data from fastapi"}');
	      res.json(result);
	    })
	    .catch((error) => {
	      console.error('Error:', error);
	    }); 
  } catch (error) {
  	console.log(error);
  	r = JSON.stringify({"status":error});
  }
  console.log('dont get here');
  // res.json(r);
});



app.listen(3010, () => {
  console.log('Server started on http://localhost:3010');
});