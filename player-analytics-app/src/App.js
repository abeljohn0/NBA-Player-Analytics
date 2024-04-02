import React, { useState, useRef } from 'react';
import {Link} from 'react-router-dom';
import Navigation from './Navigation.js';
import InjuryTable from './InjuryTable';


// export function handleLinkClick(player_name, category) {
//   console.log('link clicked for ', player_name)
//   setPlayerName(player_name); 
//   setCategory(category);
//   if (buttonRef.current) {
//     buttonRef.current.click();
//   }

  // displayAsTable(newData, handleLinkClick);
  // updatePlot(newData);
// };

function App() {
  // function handleAppLinkUpdate(player_name, category) {
  //   console.log('link clicked for ', player_name)
  // }
  
  return (
    <div>
      <Navigation/>
      <InjuryTable/>
          {/* <Routes> */}
            {/* <Route path="/" element={<App/>} /> */}
          {/* </Routes> */}
    </div>
  );
}

export default App;
