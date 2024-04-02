// PicksPage.js
import React, { useEffect, useState } from 'react';
import {displayAsTable} from './PlayerPage.js';
import Navigation from './Navigation.js';

const PicksPage = () => {
  const [noData, setNoData] = useState(false);
  // useEffect(() => {
  //   console.log('Picks page loaded');
  //   get_picks();
  // }, []);

  const get_picks = () => {
    fetch('/picks', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      },
    })
    .then(response => response.json())
    .then(data => {
      console.log(data)
      if (data.hasOwnProperty('no data')) {
        console.log('uh oh')
        setNoData(true);
      } else {
        displayAsTable(data);
        setNoData(false);
      }
    });
  };

  return (
    <div>
      <Navigation/>
      {noData ? (
        <div>
          <h1>No data yet... check back today at 9 AM!</h1>
        </div>
      ) : (
        <div id="table-container"/>
      )}
    </div>
  );
};

export default PicksPage;
