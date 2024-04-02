import React, { useState } from 'react';
import {displayAsTable} from './PlayerPage.js';


const InjuryTable = () => {
    const [team, setTeam] = useState('');
    const handleCategoryChange = (event) => {
        setTeam(event.target.value);
        fetch('/injuries', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({'team': event.target.value})
        }).then(response => response.json())
          .then(data_df => {
            console.log(data_df)
            displayAsTable(data_df);
          })
    };

    return (
            <div>
              <select onChange={handleCategoryChange}>
                  <option value="ATL">Atlanta Hawks</option>
                  <option value="BOS">Boston Celtics</option>
                  <option value="BKN">Brooklyn Nets</option>
                  <option value="CHA">Charlotte Hornets</option>
                  <option value="CHI">Chicago Bulls</option>
                  <option value="CLE">Cleveland Cavaliers</option>
                  <option value="DAL">Dallas Mavericks</option>
                  <option value="DEN">Denver Nuggets</option>
                  <option value="DET">Detroit Pistons</option>
                  <option value="GS">Golden State Warriors</option>
                  <option value="HOU">Houston Rockets</option>
                  <option value="IND">Indiana Pacers</option>
                  <option value="LAC">Los Angeles Clippers</option>
                  <option value="LAL">Los Angeles Lakers</option>
                  <option value="MEM">Memphis Grizzlies</option>
                  <option value="MIA">Miami Heat</option>
                  <option value="MIL">Milwaukee Bucks</option>
                  <option value="MIN">Minnesota Timberwolves</option>
                  <option value="NO">New Orleans Pelicans</option>
                  <option value="NY">New York Knicks</option>
                  <option value="OKC">Oklahoma City Thunder</option>
                  <option value="ORL">Orlando Magic</option>
                  <option value="PHI">Philadelphia 76ers</option>
                  <option value="PHO">Phoenix Suns</option>
                  <option value="POR">Portland Trail Blazers</option>
                  <option value="SAC">Sacramento Kings</option>
                  <option value="SA">San Antonio Spurs</option>
                  <option value="TOR">Toronto Raptors</option>
                  <option value="UTA">Utah Jazz</option>
                  <option value="WAS">Washington Wizards</option>
              </select>
              <div id="table-container"></div>
            </div>
    )
}

export default InjuryTable;