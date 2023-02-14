// risk_tables.js holds all client-side javascript
//  reference in index.html is <script src="js/risk_tables.js"></script>
// define the maximumn number of columns the correlation matrix can have
const CORR_COL_LIMIT = 100;

function showDiv(div_id) {
    document.getElementById(div_id).classList.remove("hide");
    document.getElementById(div_id).classList.add("show");
};

function hideDiv(div_id) {
    document.getElementById(div_id).classList.remove("show");
    document.getElementById(div_id).classList.add("hide");
};


// const get_var_url = "/riskdata"

const cols_conversions = {
  'symbol':'sym', 
  'position':'qty',
  'underlying':'usym',
  '*underlying':'usym',
  'close':'last',
  'stdev':'std',
  'delta':'del',
  'gamma':'gam',
  'vega':'veg',
  'rho':'rho',
  'theta':'theta',
  'unit_var':'uvar',
  'position_var':'pvar',
  'd1':'d1',
  'd5':'d5',
  'd10':'d10',
  'd15':'d15',
  'd20':'d20',
  };

function convert_cols(col){
  if (col in cols_conversions){
    return cols_conversions[col];
  }
  return col;
}

function convert_df_portfolio_col_names(row){
  var row_keys = Object.keys(row);
  ret = Object.assign(
    {}, ...row_keys.map(
      (x) => (
        {
          // [cols_conversions[x]]:row[x]
          [convert_cols(x)]:row[x]
        }
      )
    )
  );
  return ret;
}

function convert_df_portfolio(df_portfolio){
  // change the keys in each row of df_portolio to a shorter length
  var df_portfolio_new = df_portfolio.map(row=>convert_df_portfolio_col_names(row));
  
  // make all floats only 3 decimal places

  for (var i=0;i<df_portfolio_new.length;i++){
    var row = df_portfolio_new[i];
    Object.keys(row).forEach(function(k){
      var row_value = row[k];
      if (typeof(row_value)=='number'){
        new_row_value = Math.round(row_value*1000)/1000;
        df_portfolio_new[i][k] = new_row_value;
      }
    });
  }
  return df_portfolio_new;
}

function display_position(json_results,tag_id,cols_to_display,json_results_key='df_risk_all') {
  // get position data from server results
  // var df_portfolio = json_results['df_positions_all'];
  var df_portfolio = json_results[json_results_key];
  // convert the keys of that df_portfolio rows to keys that have smaller lengths, like 'symbol' will be 'sym'.
  df_portfolio = convert_df_portfolio(df_portfolio);
  // these are the shorter length keys that we will display in the datatables
  const converted_cols = cols_to_display.map(c=>convert_cols(c));
  // this is the dictionary that you pass to datatable
  var dt_cols = converted_cols.map(function(c){
    return {"data":c,"title":c,"visible":c[0]!=='_'}
  });
  
  // display datatable
  if ( ! $.fn.DataTable.isDataTable("#"+tag_id) ) {
    // // these are the shorter length keys that we will display in the datatables
    // const converted_cols = cols_to_display.map(c=>convert_cols(c));
    // // this is the dictionary that you pass to datatable
    // var dt_cols = converted_cols.map(function(c){
    //   return {"data":c,"title":c}
    // });
    // display the datatable
    $("#"+tag_id).dataTable( {
        "data": df_portfolio,
        "columns":dt_cols,
        "order": [[0, 'asc']],
        "pageLength": 10,
        "searching": false,
        "lengthChange": false,
        "info":false,
        "scrollX": true,
    } );      
  } else {
    // var dt = $("#"+tag_id).dataTable();
    // dt.fnDestroy();
    $("#"+tag_id).dataTable( {
        "data": df_portfolio,
        "columns":dt_cols,
        "order": [[0, 'asc']],
        "pageLength": 10,
        "searching": false,
        "lengthChange": false,
        "info":false,
        "scrollX": true,
        "destroy":true,
    } );      
    // var dt = $("#"+tag_id).dataTable();
    // dt.fnClearTable();
    // dt.fnAddData(df_portfolio,redraw=true);
  }        
};


function render_var_bar_plot(
  json_results,
  json_results_key='df_risk_by_underlying',
  x_col='underlying',
  y_col='position_var',
  output_div='var_plot',
  plot_type='bar',
  graph_title="VaR By Underlying",
  y_axis_title="VaR in Dollars",
  marker_color="#7e8ed9",) {
  var df = json_results[json_results_key];
  var x_arr = df.map(row=>row[x_col]);
  var y_arr = df.map(row=>row[y_col]);
  var trace1 = {
      x:x_arr,
      y: y_arr,
      type: plot_type,
      marker:{color:marker_color}
  };

  var data = [trace1];
  var lay_out_yaxis = {
    title: {
      text: y_axis_title,
    }
  };

  var layout = {
      title: graph_title,
      yaxis: lay_out_yaxis,
      // showlegend: false
  };  

  var plot_config = {responsive: true}
  Plotly.newPlot(output_div,data,layout,plot_config);  
}

function render_portfolio_stats(
  json_results)
{
  var port_var = Math.round(json_results["port_var"]*100)/100;
  port_var = port_var.toLocaleString("en-US", {style:"currency", currency:"USD"});
  var sp_dollar_equiv = Math.round(json_results["sp_dollar_equiv"]*100)/100;
  sp_dollar_equiv = sp_dollar_equiv.toLocaleString("en-US", {style:"currency", currency:"USD"});
  var delta = Math.round(json_results["delta"]*100)/100;
  delta = delta.toLocaleString("en-US", {style:"currency", currency:"USD"});
  var gamma = Math.round(json_results["gamma"]*100)/100;
  gamma = gamma.toLocaleString("en-US", {style:"currency", currency:"USD"});
  var vega = Math.round(json_results["vega"]*100)/100;
  vega = vega.toLocaleString("en-US", {style:"currency", currency:"USD"});
  var theta = Math.round(json_results["theta"]*100)/100;
  theta = theta.toLocaleString("en-US", {style:"currency", currency:"USD"});

  document.getElementById('port_var').innerHTML = "Portfolio VaR: " + port_var;
  document.getElementById('sp_dollar_equiv').innerHTML = "S&P Dollar Equivalent: " + sp_dollar_equiv;
  document.getElementById('port_delta').innerHTML = "Delta: " + delta; 
  document.getElementById('port_gamma').innerHTML = "Gamma: " + gamma; 
  document.getElementById('port_vega').innerHTML = "Vega: " + vega; 
  document.getElementById('port_theta').innerHTML = "Theta: " + theta; 
}

function display_json_results(json_results) {
  const pos_cols_to_display =  ['symbol','position','position_var'];
  const greeks_cols_to_display = ['underlying','delta','gamma','vega','theta','rho','position_var']
  const underlying_cols_to_display =  ['underlying','position_var'];
  const atm_info_cols_to_display = ['underlying','close','stdev','d1','d5','d10','d15','d20'];

  // The position, greeks and atm_info datasets always have the same columns.  Therefore,
  //  no special treatment is necessary to re-display those datatables.
  display_position(json_results,'position',pos_cols_to_display);
  render_portfolio_stats(json_results);
  render_var_bar_plot(json_results);
  display_position(
    json_results,'greeks2',greeks_cols_to_display,json_results_key='df_risk_by_underlying'
  );
  display_position(
    json_results,'atm_info',atm_info_cols_to_display,json_results_key='df_atm_info'
  );

  // The correlation matrix columns are equal to the number of underlyings, plus 1.
  //  Because the columns vary, and because the datatables api cannot re-initialize the number
  //  of columns, you need to add dummy columns to the data.  I have these column names start with
  //  the '_' character.  Ther display_position method will then make those columns NON-visible
  // Step 1: Get the columns with actual data
  var cor_matrix_cols = Object.keys(json_results['df_corr'][0]);
  // Step 2: Create the NON display column names
  // the line below is like python:
  //    range(1,lim - len(cor_matrix_cols))
  var non_display_cols = Array.from(
    {length:CORR_COL_LIMIT-cor_matrix_cols.length},(v,i)=>'_'+(i+1).toString()
  );
  // Step 3: Create null data in each roww of df_corr, for the NON-display columns
  var df_corr = json_results['df_corr'];
  var df_corr_row_indices = Array.from({length:df_corr.length},(v,i)=>i);
  for (var c of non_display_cols){
    for (var i of df_corr_row_indices){
      df_corr[i][c] = null;
    }
  }
  // Step 4: Concatenate the good columns with the NON-display columns
  cor_matrix_cols = cor_matrix_cols.concat(non_display_cols);
  // Step 5: store the new df_corr data in the json_results array
  json_results['df_corr'] = df_corr;
  // Step 6: Call display_position
  display_position(json_results,'corr_matrix',cor_matrix_cols,json_results_key='df_corr');

};

// function display_risk_tables(){
//   // get var stuff from risk_server.py
//   // fetch and display data
//   fetch_var(getfull=0)
//   .then(function(json_results){
//     // console.log(json_results);
//     display_json_results(json_results);
//   });  
// }

async function get_local_csv_file() {
  // const content = document.querySelector('#filecontent');
  const [file] = document.querySelector('input[type=file]').files;
  const reader = new FileReader();

  reader.addEventListener("load", () => {
    // this will then display a text file
    // content.innerText = reader.result;
    // change carriage return and line feed to semi-colon
    var csv_text = reader.result.replaceAll('\n',';');
    csv_text = csv_text.replaceAll('\r',';');
    // send the csv text string to express server, which will send it to python server
    showDiv('spinner');
    upload_csv_to_server(csv_text)
    .then(function(json_results){
      // console.log(json_results);
      if (json_results!==null){
        display_json_results(json_results);
      } else {
        alert('bad csv file uploaded')
      }
      hideDiv('spinner');
    });  

  }, false);

  if (file) {
    reader.readAsText(file);
  }
};

async function display_default_portfolio() {
  const response = await fetch('/default_portfolio', {
    method: 'GET',
    headers: {
        // 'Accept': 'application/json',
        'Content-Type': 'application/json',
    }        
  });
  if (response.ok) { // if HTTP-status is 200-299
    // get the response body (the method explained below)
    let json = await response.json();
    var csv_text = json['csv_text'];
    csv_text = csv_text.replaceAll('\n',';');
    csv_text = csv_text.replaceAll('\r',';');
    showDiv('spinner');
    upload_csv_to_server(csv_text)
    .then(function(json_results){
      // console.log(json_results);
      if (json_results!==null){
        display_json_results(json_results);
      } else {
        alert('bad csv file uploaded')
      }
      hideDiv('spinner');
    });  
  } else {
      alert("HTTP-Error: " + response.status);
      return null;
  }  
};

async function upload_csv_to_server(csv_text){
  const response = await fetch('/riskdata_from_csv', {
    method: 'POST',
    // method: 'GET',
    headers: {
        // 'Accept': 'application/json',
        'Content-Type': 'application/json',
    },        
    body: JSON.stringify({'data':csv_text})
  })
  if (response.ok) { // if HTTP-status is 200-299
    // get the response body (the method explained below)
    let json = await response.json();
    return json;
  } else {
      alert("HTTP-Error: " + response.status);
      return null;
  }  

    // .then((response) => response.json())
    // .then((result) => {
    //   console.log('Success:', Object.keys(result));
    //   return result;
    // })
    // .catch((error) => {
    //   console.log('Error:'+ error);
    //   return null;
    // });    
};

function initit(){
  // hideDiv('spinner');
  // display_risk_tables();
  display_default_portfolio();
}