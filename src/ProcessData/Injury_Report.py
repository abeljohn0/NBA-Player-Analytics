import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
import sqlite3
def aggregate_injury_data():
    url = 'https://www.cbssports.com/nba/injuries/'

    session = requests.Session()

    response = session.get(url)

    injury_df = pd.DataFrame(columns=['Team', 'Player Name', 'Position', 'Updated', 'Injury', ' Injury Status'])

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)

        table_wrappers = soup.find_all('div', class_='TableBaseWrapper')

        for table_wrapper in table_wrappers:
            team_name_span = table_wrapper.find('span', class_='TeamName')
            href = team_name_span.find('a')['href']
            team_abbreviation = href.split('/')[3]
            print(team_abbreviation)
            player_wrappers = table_wrapper.find_all('tr', class_='TableBase-bodyTr')
            for player_wrapper in player_wrappers:
                elems = [team_abbreviation]
                injury_elems = player_wrapper.find_all('td', class_='TableBase-bodyTd')
                for elem in injury_elems:
                    if elem.find('span'):
                        txt = elem.find_all('span')[-1].text
                    else:
                        txt = elem.text
                    elems.append(txt.strip())
                injury_df.loc[len(injury_df)] = elems

    con = sqlite3.connect("Data/games.sqlite")
    injury_df.to_sql(f"injury_report", con, if_exists="replace")
    con.close()
    print('injury report compiled successfully!')

def get_injury_data(team_name):
     con = sqlite3.connect("Data/games.sqlite")
     injuries_df = pd.read_sql_query(f"select * from \"injury_report\"", con, index_col="index")
     con.close()
    #  print(injuries_df)
     return injuries_df[injuries_df['Team'] == team_name].reset_index(drop=True)

# aggregate_injury_data()