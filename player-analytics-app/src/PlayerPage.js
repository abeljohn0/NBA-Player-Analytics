import React, { useState, useEffect } from 'react';
import {  useParams, Link  } from 'react-router-dom';
import Plotly from 'plotly.js-basic-dist';
import Navigation from './Navigation.js';
import InjuryTable from './InjuryTable';

export function displayAsTable(data) {
    // console.log(data)
    const tableContainer = document.getElementById('table-container');
    console.log('hi')

    tableContainer.innerHTML = '';
    console.log('hi')
    const table = document.createElement('table');
    const header = table.createTHead();
    const headerRow = header.insertRow(0);
  
    Object.keys(data).forEach(key => {
      const th = document.createElement('th');
      th.textContent = key;
      headerRow.appendChild(th);
    });
    // const body = table.createTBody();
    // const dataRow = body.insertRow(0);
    const row_vals = Object.values(data)[0]
    console.log(row_vals)
    const numRows = Object.keys(row_vals).length
    for (let i = 0; i < numRows; i++) {
        const dataRow = table.insertRow(-1);

        // Populate cells in each row
        Object.keys(data).forEach(category => {
            const cell = dataRow.insertCell(-1);
            console.log(data[category][i].toString());
            // console.log(data[category][i]);
            if (category == 'Player') {
              const link = document.createElement('a');
              const linkText = document.createTextNode(data[category][i].toString());
            
              link.appendChild(linkText);
              const player_name = data[category][i].toString()
              const encodedPlayerName = encodeURIComponent(player_name);
              const encodedCategory = encodeURIComponent(data['Category'][i].toString());
              link.href = `/player/${encodedPlayerName}/${encodedCategory}`;
              cell.appendChild(link);
            //   cell.textContent = data[category][i].toString();
            //   cell.innerHTML = (
            //     <Link to={`/player/${encodedPlayerName}/${encodedCategory}`} key={player_name}>
            //       {player_name}
            //     </Link>
            //   );
            }
            else {
            cell.textContent = data[category][i].toString();
            }
        });
    }
  
    for (const key in data['data']) {
      if (Object.hasOwnProperty.call(data['data'], key)) {
        const rowData = data['data'][key];
    
        const dataRow = table.insertRow(-1);
    
        Object.values(rowData).forEach(value => {
          const cell = dataRow.insertCell(-1);
          cell.textContent = value;
        });
      }
    }
    tableContainer.appendChild(table);
  };

function PlayerPage() {
//   console.log('PlayerPage component rendered.');
  const [imageUrl, setImageUrl] = useState('');
  const { playerName, category } = useParams();
  console.log(playerName)
  console.log(category)
  const data = {
    player_name: decodeURIComponent(playerName),
    category: decodeURIComponent(category)
  };
  console.log(data);
  const updatePlot = (data) => {
    // Modify this part based on your specific data structure
    const line = {
      x: Object.values(data['Date']),
      y: Object.values(data['Line']),
      type: 'scatter',
      name: 'Line',
    };

    const results = {
      x: Object.values(data['Date']),
      y: Object.values(data['Score']),
      type: 'scatter',
      name: 'Score',
    };
    const model_pred = {
      x: Object.values(data['Date']),
      y: Object.values(data['Result']),
      type: 'scatter',
      name: 'model_pred',
    };
    var data = [line, results, model_pred];

    Plotly.newPlot('myDiv', data);
  };

  fetch('/player', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data_df => {
    console.log(data_df)
    displayAsTable(data_df);
    updatePlot(data_df);
  });

  fetch('/player_photo', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(img_url => {
    console.log(img_url)
    // document.getElementById("image").src = img_url
    setImageUrl(img_url['player_photo']);
    // displayAsTable(data_df);
    // updatePlot(data_df);
  });
  return (
    <div>
      <Navigation/>
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <div id="table-container" style={{ flex: 1 }}/>
        <img src={imageUrl} id="image" style={{ position: 'absolute', top: 100, right: 20}}/>
      </div>
      <head>
	      <script src='https://cdn.plot.ly/plotly-2.29.1.min.js'></script>
      </head>
      <body>
	      <div id='myDiv'></div>
      </body>
      <InjuryTable/>
    </div>
  );
};

export default PlayerPage;