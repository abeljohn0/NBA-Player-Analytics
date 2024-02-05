from datetime import datetime
import re
import requests
import pandas as pd
from .Dictionaries import team_index_current

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive'
}

pp_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive'
}


url_team = 'https://stats.nba.com/stats/' \
      'leaguedashteamstats?Conference=&' \
      'DateFrom=10%2F01%2F{0}&DateTo={1}' \
      '&Division=&GameScope=&GameSegment=&LastNGames=0&' \
      'LeagueID=00&Location=&MeasureType=Advanced&Month=0&' \
      'OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
      'PerMode=PerGame&Period=0&PlayerExperience=&' \
      'PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season={2}' \
      '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

url_szn_games = 'https://stats.nba.com/stats/leaguegamefinder?' \
      'Conference=&DateFrom=10%2F01%2F{0}&DateTo=6%2F30%2F{1}&Division=&DraftNumber=&'\
      'DraftRound=&DraftTeamID=&DraftYear=&EqAST=&EqBLK=&EqDD=&'\
      'EqDREB=&EqFG3A=&EqFG3M=&EqFG3_PCT=&EqFGA=&EqFGM=&EqFG_PCT=&'\
      'EqFTA=&EqFTM=&EqFT_PCT=&EqMINUTES=&EqOREB=&EqPF=&EqPTS=&'\
      'EqREB=&EqSTL=&EqTD=&EqTOV=&GameID=&GtAST=&GtBLK=&GtDD=&'\
      'GtDREB=&GtFG3A=&GtFG3M=&GtFG3_PCT=&GtFGA=&GtFGM=&GtFG_PCT=&'\
      'GtFTA=&GtFTM=&GtFT_PCT=&GtMINUTES=&GtOREB=&GtPF=&GtPTS=&GtREB=&'\
      'GtSTL=&GtTD=&GtTOV=&LeagueID=00&Location=&LtAST=&LtBLK=&LtDD=&'\
      'LtDREB=&LtFG3A=&LtFG3M=&LtFG3_PCT=&LtFGA=&LtFGM=&LtFG_PCT=&'\
      'LtFTA=&LtFTM=&LtFT_PCT=&LtMINUTES=&LtOREB=&LtPF=&LtPTS=&'\
      'LtREB=&LtSTL=&LtTD=&LtTOV=&Outcome=&PORound=0&PlayerID={4}&'\
      'PlayerOrTeam={3}&RookieYear=&Season={2}&SeasonSegment=&SeasonType=Regular+Season&'\
      'StarterBench=Starters&TeamID=&VsConference=&VsDivision=&VsTeamID=&YearsExperience='

url_player_cur_game = 'https://stats.nba.com/stats/' \
       'leaguedashplayerstats?College=&'\
       'Conference=&Country=&DateFrom={0}&DateTo={0}'\
       '&Division=&DraftPick=&DraftYear=&GameScope=&'\
       'GameSegment=&Height=&LastNGames=0&LeagueID=00&'\
       'Location=&MeasureType=Base&Month=0&OpponentTeamID=0&'\
       'Outcome=&PORound=&PaceAdjust=N&PerMode=PerGame&Period=0&'\
       'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&'\
       'Season={1}&SeasonSegment=&SeasonType=Regular+Season&'\
       'ShotClockRange=&StarterBench={2}&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='

url_player_szn = 'https://stats.nba.com/stats/' \
       'leaguedashplayerstats?College=&'\
       'Conference=&Country=&DateFrom=10%2F01%2F{0}&DateTo={1}'\
       '&Division=&DraftPick=&DraftYear=&GameScope=&'\
       'GameSegment=&Height=&LastNGames=0&LeagueID=00&'\
       'Location=&MeasureType=Base&Month=0&OpponentTeamID=0&'\
       'Outcome=&PORound=&PaceAdjust=N&PerMode=PerGame&Period=0&'\
       'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&'\
       'Season={2}&SeasonSegment=&SeasonType=Regular+Season&'\
       'ShotClockRange=&StarterBench=Starters&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='

url_player_adv_szn = 'https://stats.nba.com/stats/' \
       'leaguedashplayerstats?College=&'\
       'Conference=&Country=&DateFrom=10%2F01%2F{0}&DateTo={1}'\
       '&Division=&DraftPick=&DraftYear=&GameScope=&'\
       'GameSegment=&Height=&LastNGames=0&LeagueID=00&'\
       'Location=&MeasureType=Advanced&Month=0&OpponentTeamID=0&'\
       'Outcome=&PORound=&PaceAdjust=N&PerMode=PerGame&Period=0&'\
       'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&'\
       'Season={2}&SeasonSegment=&SeasonType=Regular+Season&'\
       'ShotClockRange=&StarterBench=Starters&TeamID=0&TwoWay=0&VsConference=&VsDivision=&Weight='

url_player_details = 'https://stats.nba.com/stats/commonplayerinfo?LeagueID=&PlayerID={0}'

url_games_td = 'https://stats.nba.com/stats/scoreboardv2?DayOffset=0&GameDate={0}&LeagueID=00'



def get_json_data(url):
    raw_data = requests.get(url, headers=data_headers)
    try:
        json = raw_data.json()
    except Exception as e:
        print(e)
        return {}
    return json.get('resultSets')


def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
    return json.get('gs').get('g')


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))

def mod_szn_cols(player_df):
    new_names = {'FGM':'SZN_FGM', 'FGA':'SZN_FGA', 'FG_PCT':'SZN_FG_PCT', 'FG3M':'SZN_FG3M', 'FG3A':'SZN_FG3A', 
    'FG3_PCT':'SZN_FG3_PCT', 'FTM':'SZN_FTM', 'FTA':'SZN_FTA', 'FT_PCT':'SZN_FT_PCT', 'OREB':'SZN_OREB',
    'DREB':'SZN_DREB', 'REB':'SZN_REB', 'AST':'SZN_AST', 'TOV':'SZN_TOV', 'STL':'SZN_STL', 'BLK':'SZN_BLK',
    'BLKA':'SZN_BLKA', 'PF':'SZN_PF', 'PFD':'SZN_PFD', 'PTS':'SZN_PTS', 'PLUS_MINUS':'SZN_PLUS_MINUS', 
    'NBA_FANTASY_PTS':'SZN_NBA_FANTASY_PTS', 'DD2':'SZN_DD2', 'TD3':'SZN_TD3'}
    labels_to_keep = list(new_names.values())
    labels_to_keep.append('PLAYER_ID')
    player_df.rename(columns=new_names, inplace=True)
    return player_df[labels_to_keep]
    # player_game_df['SZN_FGM'] = player_df['FGM']
    # player_game_df['SZN_FGA'] = player_df['FGA']
    # player_game_df['SZN_FG_PCT'] = player_df['FG_PCT']
    # player_game_df['SZN_FG3M'] = player_df['FG3M']
    # player_game_df['SZN_FG3A'] = player_df['FG3A']
    # player_game_df['SZN_FG3_PCT'] = player_df['FG3_PCT']
    # player_game_df['SZN_FTM'] = player_df['FTM']
    # player_game_df['SZN_FTA'] = player_df['FTA']
    # player_game_df['SZN_FT_PCT'] = player_df['FT_PCT']
    # player_game_df['SZN_OREB'] = player_df['OREB']
    # player_game_df['SZN_DREB'] = player_df['DREB']
    # player_game_df['SZN_REB'] = player_df['REB']
    # player_game_df['SZN_AST'] = player_df['AST']
    # player_game_df['SZN_TOV'] = player_df['TOV']
    # player_game_df['SZN_STL'] = player_df['STL']
    # player_game_df['SZN_BLK'] = player_df['BLK']
    # player_game_df['SZN_BLKA'] = player_df['BLKA']
    # player_game_df['SZN_PF'] = player_df['PF']
    # player_game_df['SZN_PFD'] = player_df['PFD']
    # player_game_df['SZN_PTS'] = player_df['PTS']
    # player_game_df['SZN_PLUS_MINUS'] = player_df['PLUS_MINUS']
    # player_game_df['SZN_NBA_FANTASY_PTS'] = player_df['NBA_FANTASY_PTS']
    # player_game_df['SZN_DD2'] = player_df['DD2']
    # player_game_df['SZN_TD3'] = player_df['TD3']
    # player_game_df['MIN'] = player_df['MIN']

def add_rest_days(teams, szn_games_df, date):
    # from pudb import set_trace; set_trace()
    team_games = szn_games_df[szn_games_df['TEAM_ID'].isin(teams)]
    sorted_df = team_games.loc[szn_games_df['DATE'] < date].sort_values(by=['TEAM_ID', 'DATE'])
    last_occurence = sorted_df.groupby('TEAM_ID').last().reset_index()
    last_occurence['DAYS_REST'] = (date - last_occurence['DATE']).dt.days.astype(float)
    return last_occurence[['TEAM_ID', 'DAYS_REST']]

def add_game_info(szn_games_df, team_df, team_to_id, date):
    '''adds whether team is at home or away for a game. Also adds opponent's
       defensive rating as a datapoint'''
    subset_df = szn_games_df.loc[szn_games_df['DATE'] == date][['GAME_ID', 'TEAM_ID', 'DATE', 'MATCHUP']].copy()
    subset_df['HOME'] = 1
    temp = subset_df['MATCHUP'].str.split(" ")
    subset_df['OPP_ID'] = temp.str[2]
    subset_df['OPP_ID'] = subset_df['OPP_ID'].map(team_to_id)
    subset_df['HOME'] = temp.str[1] != '@'
    # cur_teams = team_df[team_df['TEAM_ID'].isin(temp.str[2].map(team_to_id))]
    cur_teams = team_df[['TEAM_ID', 'TEAM_DEF_RATING']].copy()
    cur_teams.rename(columns={'TEAM_ID': 'OPP_ID'}, inplace=True)
    subset_df = pd.merge(subset_df, cur_teams, on='OPP_ID', how='left')
    subset_df.rename(columns={'TEAM_DEF_RATING': 'OPP_DEF_RATING'}, inplace=True)
    return subset_df

def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games

def get_date(date_string):
    year1,month,day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')