import os
import random
import sqlite3
# import sys
import time
import pandas as pd
import pytz
from datetime import date, datetime, timedelta
from tqdm import tqdm
# sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.tools import get_json_data, to_data_frame, mod_szn_cols, add_rest_days, add_game_info, \
url_team, url_szn_games, url_player_cur_game, url_player_szn, url_player_adv_szn, url_player_details, url_games_td
from src.Utils.Dictionaries import team_to_id, pos_id
from Data.players import find_players_by_full_name
from src.Utils.Dictionaries import cat_to_metric
from src.ProcessData.Injury_Report import aggregate_injury_data

year = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
seasons = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

def get_player_id(player_name):
    player_list = find_players_by_full_name(player_name)
    player_id = ""
    for player in player_list:
        if player["full_name"] == player_name:
            player_id = player["id"]
            break
    return player_id

def generate_total_dataset(db_name='games'):
    # aggregate in dataset for all seasons
    # TODO: make a way to check if dataset has already been created for player
    con_local = sqlite3.connect(f"Data/{db_name}.sqlite")
    total_szn_df = pd.DataFrame
    for season in tqdm(seasons):
        szn_df = pd.read_sql_query(f"select * from \"{db_name}_{season}\"", con_local, index_col="index")
        if total_szn_df.empty:
            total_szn_df = szn_df
        else:
            total_szn_df = pd.concat([total_szn_df, szn_df], ignore_index=True)
    total_szn_df.to_sql(f"{db_name}_2014-2024", con_local, if_exists="replace")
    con_local.close()
    return total_szn_df

def add_player_position(total_df, db_name='games', con=None, location='2014-2024'):
    # add the positions for each player PG, SG, C, PF, etc...
    total_position = []
    uniq_players = total_df['PLAYER_ID'].unique()
    players_and_pos = {'PLAYER_ID': [], 'POSITION': []}
    for player_id in tqdm(uniq_players):
        #   uniq_players
        players_and_pos['PLAYER_ID'].append(player_id)
        player_pos = get_json_data(url_player_details.format(player_id))
        player_pos = to_data_frame(player_pos)
        time.sleep(random.uniform(0.3,0.8))
        players_and_pos['POSITION'].append(player_pos['POSITION'])
    player_id_df = pd.DataFrame(players_and_pos)
    player_id_df['POSITION'] = player_id_df['POSITION'].apply(lambda x: x[0])
    player_id_df['POSITION']= player_id_df['POSITION'].map(pos_id)
    total_df = pd.merge(total_df, player_id_df, on='PLAYER_ID', how='left')
    if con is not None:
        total_df.to_sql(f"{db_name}_{location}", con, if_exists="replace")
    else:
        return total_df

def add_player_position_per_szn(con, db_name='games'):
    total_df = pd.read_sql_query(f"select * from \"{db_name}_2014-2024\"", con, index_col="index")
    total_df = total_df.drop_duplicates(subset='PLAYER_ID')
    total_df = total_df[['PLAYER_ID', 'POSITION']]
    # from pudb import set_trace; set_trace()
    for season in tqdm(seasons):
        szn_df = pd.read_sql_query(f"select * from \"{db_name}_{season}\"", con, index_col="index")
        if 'POSITION' in szn_df.columns:
            szn_df = szn_df.drop(columns=['POSITION'])
        szn_df = pd.merge(szn_df, total_df, on='PLAYER_ID', how='left')
        szn_df.to_sql(f"{db_name}_{season}", con, if_exists="replace")

# TODO: change this to 'teams' instead of 'games'
def aggregate_data_by_season(db_name='games', player_name="", update=False, player_past_date=None):
    endpt_obj = 'T'
    player_id = ""
    if player_name != "":
        # db_name = player_name.replace(" ", "_")
        year = [2023, 2024]
        seasons = ["2023-24"]
        endpt_obj = 'P'
        player_id = get_player_id(player_name)
        if player_id == "":
            print('no player found for: ', player_name)
            return pd.DataFrame()
    else:
        year = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        seasons = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
        con = sqlite3.connect(f"Data/{db_name}.sqlite")
        print('sql db cnxn opened for: ', db_name)
        if update:
            # from pudb import set_trace; set_trace()
            year = [2023, 2024]
            seasons = ["2023-24"]
            old_season_df = pd.read_sql_query(f"select * from \"games_2023-24\"", con, index_col="index")
            # from pudb import set_trace; set_trace()
            old_season_df =  old_season_df[old_season_df['DATE'].notna()]
    count = 0
    begin_year_pointer = year[count]
    end_year_pointer = year[count + 1]
    for season in tqdm(seasons):
        begin_year_pointer = year[count]
        end_year_pointer = year[count + 1]
        # dates = []
        # else:
        szn_games = get_json_data(url_szn_games.format(begin_year_pointer, end_year_pointer, season, endpt_obj, player_id))
        szn_games_df = to_data_frame(szn_games)
        # organize season by date
        szn_games_df['DATE'] = pd.to_datetime(szn_games_df['GAME_DATE'])
        dates = szn_games_df['DATE'].drop_duplicates().reset_index(drop=True)
        szn_games_df = szn_games_df.sort_values(by='DATE')
        szn_games_df = szn_games_df.reset_index(drop=True)
        season_df = pd.DataFrame
        # dummy date for debugging
        # datetime_string = '2015-02-11 00:00:00'
        # date = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
        # dates = [date]
        # dates = [dates[0]]
        if player_name != "":
            if player_past_date is None:
                dates = [pd.to_datetime(datetime.now().date())]
            else:
                # date = datetime.strptime(player_past_date, "%m-%d-%Y")
                # from pudb import set_trace; set_trace()
                dates = [pd.to_datetime(player_past_date)]
        if update:
            # from pudb import set_trace; set_trace()
            date = old_season_df.sort_values(by='DATE').tail(1)['DATE']
            if str(date.values[0]) == str(dates[0]):
                return season_df
            date_index = dates[dates==date.values[0]].index[0]
            dates = dates[:date_index]
        for date in tqdm(dates):
            # from pudb import set_trace; set_trace()
            # use this date for certain URL calls
            prev_date = date - timedelta(days=1)
            general_data = get_json_data(url_team.format(begin_year_pointer, prev_date, season))
            team_df = to_data_frame(general_data).drop(columns=['TEAM_NAME', 'GP', 'W', 'L', 'MIN'])
            team_df = team_df.add_prefix("TEAM_")
            team_df.rename(columns={'TEAM_TEAM_ID': 'TEAM_ID'}, inplace=True)
            time.sleep(random.uniform(0.3, 0.7))
            
            player_data = get_json_data(url_player_szn.format(begin_year_pointer, prev_date, season))
            # season averages for player
            player_df = to_data_frame(player_data)
            # TODO: replace mod_szn_cols w this implementation... player_df.add_prefix("SZN_")
            player_df = mod_szn_cols(player_df)
            time.sleep(random.uniform(0.3, 0.7))

            player_adv_data = get_json_data(url_player_adv_szn.format(begin_year_pointer, prev_date, season))
            # advanced stats szn avgs - OFF_RTG, PACE, etc.
            player_adv_df = to_data_frame(player_adv_data)
            time.sleep(random.uniform(0.3, 0.7))

            if player_name == "":
                gameday_player_data = get_json_data(url_player_cur_game.format(date, season, 'Starters'))
                gameday_player_df = to_data_frame(gameday_player_data)
                # scoring metrics for a given game
                player_game_df = gameday_player_df[['PLAYER_ID', 'TEAM_ID', 'AGE', 'FGM', 'FGA', 'FG_PCT',  'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 
                'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 
                'NBA_FANTASY_PTS', 'DD2', 'TD3']].copy()
                time.sleep(random.uniform(0.3, 0.7))
                adv_drop = ['PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'AGE',  'FGA', 'FG_PCT', 'FGM']
            else:
                # from pudb import set_trace; set_trace()
                gameday_player_data = get_json_data(url_player_cur_game.format(date, season, 'Starters'))
                gameday_player_df = to_data_frame(gameday_player_data)
                player_game_df = pd.DataFrame()
                player_game_df['TEAM_ID'] = szn_games_df['TEAM_ID'].unique().copy()
                player_game_df['PLAYER_ID'] = player_id
                gameday_data = get_json_data(url_games_td.format(date))
                gameday_df = to_data_frame(gameday_data)
                adv_drop = ['PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGA', 'FG_PCT', 'FGM']
                if (gameday_df['HOME_TEAM_ID'].isin(player_game_df['TEAM_ID']).any()):
                    player_game_df['HOME'] = 1
                    opp_id = gameday_df.loc[gameday_df['HOME_TEAM_ID']==player_game_df['TEAM_ID'].values[0], 'VISITOR_TEAM_ID']
                else:
                    player_game_df['HOME'] = 0
                    opp_id = gameday_df.loc[gameday_df['VISITOR_TEAM_ID']==player_game_df['TEAM_ID'].values[0], 'HOME_TEAM_ID']
                # from pudb import set_trace; set_trace()
                try:
                    player_game_df['OPP_DEF_RATING'] = team_df.loc[team_df['TEAM_ID'] == opp_id.values[0]]['TEAM_DEF_RATING'].copy().values[0]
                except:
                    # from pudb import set_trace; set_trace()
                    print("no data for this player")
            # from pudb import set_trace; set_trace()
            # TODO: drop FGA, FG_PCT, and FGM from this... maybe FG3M as well I think that may be causing an issue??? And everything else lol
            player_adv_df = player_adv_df.drop(columns=adv_drop)
            rest_days_df = add_rest_days(player_game_df['TEAM_ID'].drop_duplicates(), szn_games_df, date)
            player_game_df = pd.merge(player_game_df, player_df, on='PLAYER_ID', how='left')
            player_game_df = pd.merge(player_game_df, player_adv_df, on='PLAYER_ID', how='left')
            player_game_df = pd.merge(player_game_df, team_df, on='TEAM_ID', how='left')
            player_game_df = pd.merge(player_game_df, rest_days_df, on='TEAM_ID', how='left')
            if player_name == "":
                subset_df = add_game_info(szn_games_df, team_df, team_to_id, date)
                player_game_df = pd.merge(player_game_df, subset_df, on='TEAM_ID', how='left')
            else:
                player_game_df = add_player_position(player_game_df)
                player_dates = szn_games_df[szn_games_df['PLAYER_ID'] == player_game_df['PLAYER_ID'][0]]['DATE']
                date_idx = player_dates.index[player_dates == date]
                last_date_idx = player_dates.tail().index[-1]
                try:
                    player_game_df['GAMES_FROM_PRESENT'] = last_date_idx - date_idx
                except:
                    # from pudb import set_trace; set_trace()
                    print('^^^INVALIDATE THIS ENTRY, unless future data lol^^^')
                    player_game_df['GAMES_FROM_PRESENT'] = 0
                return player_game_df
            if season_df.empty:
                season_df = player_game_df
            else:
                season_df = pd.concat([season_df, player_game_df], ignore_index=True)
        if update:
            # old_season_df['DATE'] = pd.to_datetime(old_season_df['DATE'])
            # from pudb import set_trace; set_trace()
            season_df['DATE'] = season_df['DATE'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            season_df = add_player_position(season_df, db_name, location=season)
            season_df = pd.concat([old_season_df, season_df], ignore_index=True)
            # season_df = season_df.dropna()
            season_df['DATE'] = pd.to_datetime(season_df['DATE'], format="%Y-%m-%d %H:%M:%S")
            season_df = season_df.sort_values(by='DATE')
            print('updated dataset to ', dates.tail()[0])
            season_df.to_sql(f"{db_name}_{season}", con, if_exists="replace")
            con.close()
            return season_df, dates
        season_df.to_sql(f"{db_name}_{season}", con, if_exists="replace")
        from pudb import set_trace; set_trace()
        count += 1
    total_df = generate_total_dataset(db_name)
    add_player_position(total_df=total_df, db_name=db_name, con=con)
    add_player_position_per_szn(con, db_name)
    con.close()
    return total_df

def get_player_results(date, category='all', specific_player_df=None, get_accuracy=False):
    con = sqlite3.connect("Data/games.sqlite")
    cor = 0
    total = 0
    if specific_player_df is None:
        try:
            players_df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
        except:
            print('no data for this date... please enter a date AFTER Jan 1st and BEFORE current date')
            players_df = pd.DataFrame()
    else:
        players_df = specific_player_df
    results_df = to_data_frame(get_json_data(url_player_cur_game.format(date, '2023-24', '')))
    if results_df.empty or players_df.empty:
        return pd.DataFrame()
    time.sleep(random.uniform(0.3, 0.7))
    for index, row in players_df.iterrows():
        # from pudb import set_trace; set_trace()
        if category != 'all' and row['Category'] != category:
            continue
        player_game_results = results_df[results_df['PLAYER_NAME'] == row['Player']]
        try:
            cat = cat_to_metric[row['Category']]
            if len(cat) > 1:
                final_result = 0
                for x in cat:
                    final_result += float(player_game_results[x])
            else:
                final_result = float(player_game_results[cat[0]])
        except:
            # from pudb import set_trace; set_trace()
            # print(f'error: {row["Player"]} not found')
            continue
        total += 1
        correct_pred = 'N/A'
        if final_result < float(row['Line']):
            correct_pred = 'UNDER'
        if final_result > float(row['Line']):
            correct_pred = 'OVER'
        if correct_pred == row['Prediction']:
            cor += 1
            decision = 'CORRECT'
        else:
            if correct_pred == 'N/A':
                decision = 'DISCOUNTED'
                # if row['Prob'] > thresholds[row['Category']]:
                total -= 1
            else:
                decision = 'INCORRECT'
        # print(f'{Fore.MAGENTA}{row["Player"]}{Style.RESET_ALL}', '\nLine: ', row['Line'], ' Category: ', row['Category'], '\nProb: ', row['Prob'], '\nPrediction: ', row['Result'], ' Result: ', final_result, decision)
        players_df.loc[index, 'Outcome'] = decision
        players_df.loc[index, 'Score'] = final_result
    con.close()
    if get_accuracy:
        return cor, total, cor/total, players_df
    return players_df


def update_df():
    _, dates = aggregate_data_by_season(update=True)
    con = sqlite3.connect("Data/games.sqlite")
    pp_df = pd.read_sql_query(f"select * from \"pp_2023-24\"", con, index_col="index")
    for date in dates:
        print(date)
        results_df = get_player_results(date)
        pp_df = pd.concat([pp_df, results_df]).drop_duplicates().reset_index(drop=True)
    pp_df.to_sql(f"pp_2023-24", con, if_exists="replace")
    con.close()
    aggregate_injury_data()
#TODO: add a reset a season functionality

# update_df()
# aggregate_data_by_season()
# con = sqlite3.connect("../../Data/games.sqlite")
# total_df = pd.read_sql_query(f"select * from \"games_2014-2024\"", con, index_col="index")
# # from pudb import set_trace; set_trace()
# add_player_position(total_df=total_df, db_name='games', con=con)
# add_player_position_per_szn(con)
# con.close()
# season_df = pd.read_sql_query(f"select * from \"games_2023-24\"", con, index_col="index")
# dataset = 'games_2014-15'
# # total_df = generate_total_dataset()
# total_df = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
# add_player_position(season_df, db_name='games', location='2023-24', con=con)
# con.close()
