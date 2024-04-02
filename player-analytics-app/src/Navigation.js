import React, { useState } from 'react';
import './Navigation.css';
import {Link, useNavigate} from 'react-router-dom';
const Navigation = () => {
    const [player_name, setPlayerName] = useState('');
    const [category, setCategory] = useState('all');  
    const navigate = useNavigate();

    const handleInputChange = (event) => {
        setPlayerName(event.target.value);
      };
    
      const handleCategoryChange = (event) => {
        setCategory(event.target.value);
      };
    
      const handleSubmit = () => {
        console.log('arrived at submit button!!!')
        // const data = {
        //   player_name: player_name,
        //   category: category
        // };
        // fetch('/player', {
        //   method: 'POST',
        //   headers: {
        //     'Content-Type': 'application/json'
        //   },
        //   body: JSON.stringify(data)
        // })
        // .then(response => response.json())
        // .then(data => {
        //   console.log('hi')
        //   console.log(data)
        //   displayAsTable(data);
        //   updatePlot(data);
        // });
        const encodedPlayerName = encodeURIComponent(player_name);
        const encodedCategory = encodeURIComponent(category);
        // link.href = '/';
        // cell.appendChild(link);
        console.log(`/player/${encodedPlayerName}/${encodedCategory}`)
        navigate(`/player/${encodedPlayerName}/${encodedCategory}`);
        // return(
        //     <Link to={`/player/${encodedPlayerName}/${encodedCategory}`}/>
        // )
      };

    return (
      <div>
        <input type="text" placeholder="Enter basketball player's name" onChange={handleInputChange} />
        <select onChange={handleCategoryChange}>
            <option value="all">All</option>
            <option value="Points">Points</option>
            <option value="Rebounds">Rebounds</option>
            <option value="Assists">Assists</option>
            <option value="Pts+Rebs+Asts">Pts+Rebs+Asts</option>
            <option value="Pts+Rebs">Pts+Rebs</option>
            <option value="Pts+Asts">Pts+Asts</option>
            <option value="Rebs+Asts">Rebs+Asts</option>
            <option value="Fantasy Score">Fantasy Score</option>
            <option value="Defensive Rebounds">Defensive Rebounds</option>
            <option value="Offensive Rebounds">Offensive Rebounds</option>
            <option value="3-PT Attempted">3-PT Attempted</option>
            <option value="Free Throws Made">Free Throws Made</option>
            <option value="FG Attempted">FG Attempted</option>
            <option value="3-PT Made">3-PT Made</option>
            <option value="Blocked Shots">Blocked Shots</option>
            <option value="Steals">Steals</option>
            <option value="Turnovers">Turnovers</option>
            <option value="Blks+Stls">Blks+Stls</option>
        </select>
        <button onClick={handleSubmit} disabled={player_name.trim() === ''}>Submit</button>
        <div className="navigation">
        <button>
          <Link to="/picks">Go to Today's Picks</Link>
        </button>
      </div>
      </div>
    )
}

export default Navigation;