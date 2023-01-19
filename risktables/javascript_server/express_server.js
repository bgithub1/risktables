const express = require('express');
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

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

app.get('/riskdata', (req, res) => {
  const get_var_url = "http://localhost:8555/get_var";
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


app.listen(3010, () => {
  console.log('Server started on http://localhost:3010');
});