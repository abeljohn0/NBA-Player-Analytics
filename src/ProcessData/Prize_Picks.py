import json
import math
import pandas as pd
import numpy as np
import sqlite3
import time
import random
# import sys
import os
from scipy.stats import norm, gaussian_kde
from datetime import datetime, timedelta
# sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.ProcessData.Get_Data import aggregate_data_by_season, update_df, get_player_id
from src.TrainModels.NN_Model_Pytorch import run_eval
from src.Utils.tools import url_player_cur_game, get_json_data, to_data_frame
from src.Utils.Dictionaries import cat_to_metric, confidence_intervals, optimal_gates
from itertools import combinations
from tqdm import tqdm


def generate_date_range(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, '%m-%d-%Y')
    end_date = datetime.strptime(end_date_str, '%m-%d-%Y')

    date_list = [start_date.strftime('%m-%d-%Y')]

    current_date = start_date + timedelta(days=1)
    while current_date <= end_date:
        date_list.append(current_date.strftime('%m-%d-%Y'))
        current_date += timedelta(days=1)

    return date_list

def create_total_pp_dset(dates, selection_type='tail', odds_type='standard'):
    con = sqlite3.connect("Data/games.sqlite")
    total_df = pd.DataFrame()
    for date in tqdm(dates):
        df = selections_by_threshold(get_player_results(date), selection_type, odds_type=odds_type)
        # df = get_player_results(date)
        # from pudb import set_trace; set_trace()
        if total_df.empty:
            total_df = df
        else:
            total_df = pd.concat([total_df, df], ignore_index=True)
    # from pudb import set_trace; set_trace()
    # print(len(total_df['Rebounds']))
    if odds_type == 'standard':
        total_df.to_sql(f"pp_preds_total_{selection_type}", con, if_exists="replace")
        total_df.to_csv(f'src/ProcessData/PrizePicksOdds/all_preds_made_{selection_type}.csv')
    con.close()
    print(f'curating {odds_type} dset for: ', selection_type, f', of size {len(total_df)}')
    return total_df

def analyze_picks(df=None, selection_type='tail', odds_type='standard'):
    con = sqlite3.connect("Data/games.sqlite")
    if df is None:
        df = pd.read_sql_query(f"select * from \"pp_preds_total_{selection_type}\"", con, index_col="index")
    # from pudb import set_trace; set_trace()
    df['Margin'] = (df['Score'] - df['Result']).abs()
    if odds_type == 'standard':
        df[['Player', 'Category', 'Outcome']].value_counts().unstack().fillna(0).to_csv(f'src/ProcessData/PrizePicksOdds/Player_Picks_Outcomes_{selection_type}.csv')
        print('mean: ', df.groupby(['Outcome', 'Category'])['Margin'].mean())
        print('median: ', df.groupby(['Outcome', 'Category'])['Margin'].median())
        print('stev: ', df.groupby(['Outcome', 'Category'])['Margin'].std())
        print(df.groupby('Category')['Outcome'].value_counts(normalize=True))
    else:
        print(df.groupby('Prediction')['Outcome'].value_counts())
        print(df[['Threshold', 'Prediction', 'Outcome']][df['Threshold']==0].value_counts().unstack())
    # print(df[df['Player'] == 'Devin Vassell'][['Prediction', 'Outcome']].value_counts().unstack().fillna(0))
    con.close()

def to_data_frame_pp(data):
    try:
        data_list = data['data']
        player_ids = data['included']
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data_list), pd.DataFrame(player_ids)

def process_row(row, szn_df):
    player_id = get_player_id(row['Player'])
    date = pd.to_datetime(row['Date'])
    szn_df['DATE'] = pd.to_datetime(szn_df['DATE'])
    metrics = szn_df[(szn_df['PLAYER_ID'] == player_id) & (szn_df['DATE'] < date)][cat_to_metric[row['Category']]].sum(axis=1)
    stev = metrics.std()
    row['stev'] = stev
    # mu = metrics.mean()
    # zscore_pred = (row['Result'] - mu)/stev
    # pred_prob = norm.cdf(zscore_pred)
    # zscore_line = (row['Line'] - mu)/stev
    # line_prob = norm.cdf(zscore_line)
    # from pudb import set_trace; set_trace()
    # row['norm_prob'] = np.abs(pred_prob-line_prob)
    # if len(metrics) > 15:
    #     kde = gaussian_kde(metrics, bw_method='silverman')
    #     pred_prob = kde.evaluate(row['Result'])[0]
    #     line_prob = kde.evaluate(row['Line'])[0]
    # # print('for pred', pred_prob)
    # # print('for line', line_prob)
    #     row['norm_prob'] = pred_prob-line_prob
    # else:
    #     row['norm_prob'] = 0
    return row

def get_stev(df):
    con = sqlite3.connect("Data/games.sqlite")
    szn_df = pd.read_sql_query(f"select * from \"games_2023-24\"", con, index_col="index")
    # df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
    # df['norm_prob'] = -1
    # for x in df:
    x = df.apply(lambda row: process_row(row, szn_df), axis=1)
    # from pudb import set_trace; set_trace()
    con.close()
    return x


def load_prize_picks(date, category='all', future=False):
    con = sqlite3.connect("Data/games.sqlite")
    # season_df = pd.read_sql_query(f"select * from \"games_2023-24\"", con, index_col="index")
    if category == 'all':
        cats = ['Fantasy Score', 'Defensive Rebounds','Offensive Rebounds', 
'3-PT Attempted', 'Free Throws Made', 'FG Attempted', '3-PT Made', 
'Blocked Shots', 'Steals', 'Turnovers', 'Rebs+Asts', 'Blks+Stls', 'Pts+Rebs+Asts', 'Pts+Rebs', 'Pts+Asts', 'Points', 'Rebounds','Assists']
    else:
        cats = [category]
    df = pd.DataFrame(columns=['Category', 'Player','Line', 'Prediction', 'Prob', 'Result', 'Odds', 'Threshold', 'Date'])
    pp_player_df = pd.DataFrame()
    with open(f'src/ProcessData/PrizePicksOdds/prizepicks_{date}.json', 'r') as file:
    # Load the JSON data into a variable
        data = json.load(file)
    data_df, player_df = to_data_frame_pp(data)
    # df_player_ids = [get_player_id(player) for player in players]
    for cat in cats:
        # from pudb import set_trace; set_trace()
        player_ids = [(data_df['relationships'][d]['new_player']['data']['id'], data_df['attributes'][d]['line_score'], data_df['attributes'][d]['odds_type'])
                    for d in range(len(data_df['attributes'])) 
                    if data_df['attributes'][d]['stat_type'] == cat]
        players_and_scores_and_type = [(player_df['attributes'][player_df.loc[player_df['id'] == x[0]].index[0]]['display_name'], x[1], x[2])
                for x in player_ids]
        for player, score, odds_type in players_and_scores_and_type:
            # if player == 'OG Anunoby':
            #     from pudb import set_trace; set_trace()
            print("prediction for: ", player)
            player_id = get_player_id(player)
            # player_identifier = season_df[season_df['DATE'] == date & season_df['PLAYER_ID'] == player_id & season_df['CATEGORY'] == cat]
            if not pp_player_df.empty and player_id in pp_player_df['PLAYER_ID'].values:
                pp_input = pp_player_df.loc[pp_player_df['PLAYER_ID'] == player_id]
                print('speed up for:', player)
            else:
                pp_input = aggregate_data_by_season(player_name=player, player_past_date=date)
                if 'OPP_DEF_RATING' not in pp_input.columns:
                    continue
                pp_player_df = pd.concat([pp_player_df, pp_input], ignore_index=True)
                
            prob, result, threshold = run_eval(pp_input, score, cat, future=future)
            if prob >= 0:
                prediction = 'OVER' if result > score else 'UNDER'
                entry = {'Category': cat, 'Player': player, 'Line': score, 'Prediction': prediction, 'Prob': prob, 'Result': result, 'Odds': odds_type, 'Threshold': threshold,'Date': date}
                print(entry)
                df.loc[len(df)] = entry
    df = df.sort_values(by='Prob').reset_index(drop=True)
    # print(df)
    if category != 'all':
        old_df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
        df = pd.concat([old_df, df], ignore_index=True)
        df = df.sort_values(by='Prob').reset_index(drop=True)
    df.to_sql(f"pp_preds_{date}", con, if_exists="replace")
    con.close()

def display_prize_picks(date, num_picks, category='all', sampling_type='total'):
    print(f'{Fore.CYAN}{Style.BRIGHT}{category}{Style.RESET_ALL}')
    con = sqlite3.connect("Data/games.sqlite")
    df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
    num_picks_all = [3, 4, 5, 6, num_picks]
    # from pudb import set_trace; set_trace()
    if category != 'all':
        df = df[df['Category'] == category]
    all_preds_df = pd.DataFrame()
    if sampling_type == 'random':
        for num_idx in range(len(num_picks_all)):
            picks = selections_by_threshold(df, sampling_type, num_picks=num_picks_all[num_idx])
            if num_idx == len(num_picks_all) - 1:
                # from pudb import set_trace; set_trace()
                picks['bet_type'] = 'combo'
            else:
                picks['bet_type'] = str(num_picks_all[num_idx])
            # print(picks)
            all_preds_df = pd.concat([picks, all_preds_df], ignore_index=True)
        df = all_preds_df
        get_player_results(date, category='all', specific_player_df=all_preds_df)
        print(all_preds_df)
    else:
        df = df.sort_values(by='Prob').reset_index(drop=True)
        print(df.tail(30))
    con.close()
    return df

def evaluate_prize_picks(date, category='all', threshold=None, threshold2=30, specific_player_df=None):
    con = sqlite3.connect("Data/games.sqlite")
    cor = 0
    total = 0
    accuracy = 0
    player_outcome = {}
    if specific_player_df is None:
        try:
            players_df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
            players_df = players_df[players_df['Odds'] == 'standard']
        except:
            print('no data for this date... please enter a date AFTER Jan 1st and BEFORE current date')
    else:
        players_df = specific_player_df
    results_df = to_data_frame(get_json_data(url_player_cur_game.format(date, '2023-24', '')))
    time.sleep(random.uniform(0.3, 0.7))
    players_df = players_df.sort_values(by='Prob').reset_index(drop=True)
    for index, row in players_df.iterrows():
        if category != 'all' and row['Category'] != category:
            continue
        if row['Player'] == 'Nicolas Claxton':
            row['Player'] = 'Nic Claxton'
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
        if threshold is not None:
            if row['Threshold'] >= threshold and row['Threshold'] <= threshold2:
                total+= 1
        correct_pred = 'N/A'
        if final_result < float(row['Line']):
            correct_pred = 'UNDER'
        if final_result > float(row['Line']):
            correct_pred = 'OVER'
        if correct_pred == row['Prediction']:
            decision = f'{Fore.GREEN}CORRECT{Style.RESET_ALL}'
            if threshold is not None:
                if row['Threshold'] >= threshold and row['Threshold'] <= threshold2:
                    cor += 1
        else:
            if correct_pred == 'N/A':
                decision = f'DISCOUNTED'
                if threshold is not None:
                    if row['Prob'] == threshold:
                        total -= 1
            else:
                decision = f'{Fore.RED}INCORRECT{Style.RESET_ALL}'
        if threshold is None:
            print(f'{Fore.MAGENTA}{row["Player"]}{Style.RESET_ALL}', '\nLine: ', row['Line'], ' Category: ', row['Category'], '\nProb: ', row['Prob'], '\nPrediction: ', row['Result'], ' Result: ', final_result, decision)
    con.close()
    if total > 0:
        accuracy = cor/total
    return accuracy, cor, total, player_outcome


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

def load_display_evaluate_pp(date, category):
    load_prize_picks(date, category)
    display_prize_picks(date, category)
    accuracy, _, _, _ = evaluate_prize_picks(date, category)
    print(f'accuracy on {date}', accuracy)

def load_display_future_data(date, category, num_picks, sampling_type='random', odds_type='standard',display_only=True):
    num_picks_all = [3, 4, 5, 6, num_picks]
    if not display_only:
        update_df()
        load_prize_picks(date, category, future=True)
        # display_prize_picks(date, category=category)
    # print(sys.path[0])
    con = sqlite3.connect("Data/games.sqlite")
    df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
    # df = get_stev(df)
    all_preds_df = pd.DataFrame()
    if sampling_type == 'random':
        for num_idx in range(len(num_picks_all)):
            picks = selections_by_threshold(df, selection_type=sampling_type, num_picks=num_picks_all[num_idx])
            if num_idx == len(num_picks_all) - 1:
                # from pudb import set_trace; set_trace()
                picks['bet_type'] = 'combo'
            else:
                picks['bet_type'] = str(num_picks_all[num_idx])
            # print(picks)
            all_preds_df = pd.concat([picks, all_preds_df], ignore_index=True)
        picks = picks[['Player', 'Prediction']].values.tolist()
    all_preds_df = selections_by_threshold(df, selection_type=sampling_type, odds_type=odds_type)
    if sampling_type == 'total':
        if odds_type == 'demon':
            print(all_preds_df.tail(30))
        else:
            print(all_preds_df)
        all_preds_df.to_csv(f'src/ProcessData/PrizePicksOdds/mypicks_{date}.csv', index=False)
        con.close()
        return all_preds_df
    if sampling_type == 'tail':
        picks = all_preds_df[['Player', 'Category', 'Prediction']].values.tolist()
    picks = list(combinations(list(picks), 2))
    print(all_preds_df)
    # from pudb import set_trace; set_trace()
    # all_preds_df.to_csv(f'src/PrizePicksOdds/mypicks_{date}.csv', index=False)
    for pick in picks:
        print(pick)
    con.close()

# DEPRECATED
def display_evaluate_pp(date, category, num_picks, threshold=None, sampling_type='threshold'):
    # from pudb import set_trace; set_trace()
    df = display_prize_picks(date, num_picks, category)
    if sampling_type == 'threshold':
        get_player_results(date, category='all', specific_player_df=df)
    else:
        if category == 'all':
            evaluate_prize_picks(date, category=category, threshold=threshold, specific_player_df = df)
        else:
            evaluate_prize_picks(date, category=category, threshold=confidence_intervals[cat_to_metric[category]][threshold])

def aggregate_preds_across_intervals(dates):
    con = sqlite3.connect("Data/games.sqlite")
    best_thresholds = {}
    # from pudb import set_trace; set_trace()
    cat_bounds = {'Pts+Asts': (3.5, 5.5), 'Rebs+Asts': (0., 10), 'Blks+Stls': (0., 10), 'Points': (2.5, 4), 'Assists': (0, 5.5), 'Fantasy Score': (2.5, 4.0),  'Rebounds': (0, 5.5), 'Defensive Rebounds': (0., 5.5),'Offensive Rebounds': (0., 5.5),     
        '3-PT Attempted': (0., 5.5), 'Free Throws Made': (0., 5.5), 'FG Attempted': (0., 10), '3-PT Made': (0., 5.5), 
        'Blocked Shots': (0., 5.5), 'Steals': (0., 5.5), 'Turnovers': (0., 5.5), 'Pts+Rebs+Asts': (4.0, 6.0), 'Pts+Rebs': (2.5, 5)}
    cat_to_threshold = {}
    for cat in cat_bounds.keys():
        print(f'{Fore.RED}{Style.BRIGHT}{cat}{Style.RESET_ALL}')
        best_cat_threshold = 0.5
        best_accuracy = 0
        threshold_acc_map = {}
        for threshold in np.arange(cat_bounds[cat][0], cat_bounds[cat][1]+0.5, 0.5):
            if cat_bounds[cat][0] == 0:
                break
            else:
                each_day_array = []
                cor_sum = 0
                total_sum = 0
                for date in dates:
                        df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
                        if cat in df['Category'].unique():
                            accuracy, cor, total, _ = evaluate_prize_picks(date, cat, threshold=threshold, threshold2=cat_bounds[cat][1])
                            each_day_array.append(accuracy)
                            cor_sum += cor
                            total_sum += total
                if total_sum > 0:
                    total_accuracy =  cor_sum/total_sum
                    threshold_acc_map[threshold] = (total_accuracy, total_sum)
                    print("Threshold: ", threshold, " ", each_day_array, " ", total_accuracy, "sample size: ", total_sum)
                if total_accuracy > best_accuracy:
                    best_accuracy = total_accuracy
                    best_cat_threshold = (threshold, best_accuracy, total_sum)
        if '+' not in cat:
            cat = cat_to_metric[cat][0]
        best_thresholds[cat] = best_cat_threshold
        cat_to_threshold[cat] = threshold_acc_map
    print(cat_to_threshold)
    return best_thresholds

# con.close()
# def test_stats(date, category):
#     if category != 'all':

def selections_by_threshold(df, selection_type='random', odds_type='standard',num_picks=6):
    all_preds_df = pd.DataFrame()
    # demon_df = df[df['Odds'] == 'goblin']
    # print(len(demon_df))
    # demon_df = demon_df[demon_df['Prediction'] == 'OVER']
    # print(len(demon_df))
    df = df[df['Odds'] == odds_type]
    if odds_type != 'standard':
        df = df.sort_values(by='Prob', ascending=False).reset_index(drop=True)
        if selection_type == 'tail':
            return df.tail(num_picks)
        return df
    else:
        for cat in optimal_gates.keys():
            lower = optimal_gates[cat][0]
            upper = optimal_gates[cat][1]
            updated_df = df[df['Category'] == cat]
            # from pudb import set_trace; set_trace()
            if odds_type == 'standard':
                updated_df = updated_df[(updated_df['Threshold'] >= lower) & (updated_df['Threshold'] <= upper)]
            all_preds_df = pd.concat([updated_df, all_preds_df], ignore_index=True)
    # print('size of dset: ', len(all_preds_df))
    all_preds_df = get_stev(all_preds_df)
    if selection_type == 'total':
        all_preds_df  = all_preds_df[all_preds_df['Player'] != 'Nicolas Claxton']
        # all_preds_df = all_preds_df.sort_values(by='norm_prob', ascending=False).reset_index(drop=True).drop_duplicates(subset='Player', keep='first')
        picks = all_preds_df.sort_values(by='Threshold', ascending=False).reset_index(drop=True)
        # .drop_duplicates(subset='Player', keep='first')
        # picks = all_preds_df.sort_values(by='Prob').reset_index(drop=True)
    if selection_type == 'random':  
        try:
            picks = all_preds_df.drop_duplicates(subset='Player', keep='first').sample(n=num_picks)
        except:
            print('not enough samples')
            picks = all_preds_df.drop_duplicates(subset='Player', keep='first')
    if selection_type == 'tail':
        # all_preds_df = get_stev(all_preds_df)
        # all_preds_df  = all_preds_df[all_preds_df['Player'] != 'Nicolas Claxton']
        # all_preds_df = all_preds_df.sort_values(by='norm_prob', ascending=False).reset_index(drop=True).drop_duplicates(subset='Player', keep='first')
        # print(all_preds_df)
        all_preds_df = all_preds_df.sort_values(by='stev', ascending=False).reset_index(drop=True).drop_duplicates(subset='Player', keep='last')
        # all_preds_df = all_preds_df.sort_values(by='Prob', ascending=False).reset_index(drop=True).drop_duplicates(subset='Player', keep='last')
        picks = all_preds_df.tail(num_picks)
    return picks

def simulate_profits(dates, num_picks, down_payment, sampling_type='random'):
    con = sqlite3.connect("Data/games.sqlite")
    running_profit = 0
    value = 0
    balance_by_day = []
    for date in range(len(dates)):
        df = pd.read_sql_query(f"select * from \"pp_preds_{dates[date]}\"", con, index_col="index")
        df = df.sort_values(by='Prob').reset_index(drop=True)
        picks = selections_by_threshold(df, sampling_type, num_picks)
        players_df = get_player_results(dates[date], category='all', specific_player_df=picks)
        print(players_df)
        player_matches = list(combinations(list(players_df['Outcome'].values), 2))
        correct = 0
        incorrect = 0
        discounted = 0
        for pairs in player_matches:
            if pairs[0] == 'CORRECT' and pairs[1] == 'CORRECT':
                correct += 1
            elif pairs[0] == 'DISCOUNTED' or pairs[1] == 'DISCOUNTED':
                discounted += 1
            elif not isinstance(pairs[0], str) and np.isnan(pairs[0]) or not isinstance(pairs[1], str) and np.isnan(pairs[1]):
                discounted += 1
            elif pairs[0] == 'INCORRECT' or pairs[1] == 'INCORRECT':
                incorrect += 1
        profit = down_payment*correct*3
        loss = down_payment*math.comb(num_picks, 2)
        refund = down_payment*discounted
        running_profit += (profit-loss+refund)
        balance_by_day.append((profit-loss+refund))
        # print('the day\'s profit:', profit-loss+refund)
    con.close()
    return running_profit, balance_by_day

def daily_accuracy_check(date, selection_type, num_picks=6, category='all', odds_type='standard', show_acc=False):
    con = sqlite3.connect("Data/games.sqlite")
    try:
        df = pd.read_sql_query(f"select * from \"pp_preds_{date}\"", con, index_col="index")
    except:
        print(date, 'OMMITTED')
        return 0, 0, 0, pd.DataFrame()
    pruned_df = selections_by_threshold(df, selection_type, odds_type, num_picks)
    # pruned_df = get_stev(pruned_df)
    # pruned_df = pruned_df.sort_values(by='norm_prob').reset_index(drop=True)
    # print(pruned_df.tail(50))
    try:
        cor, total, acc, marked_df = get_player_results(date, category=category, specific_player_df=pruned_df, get_accuracy=True)
    except:
        print(f'{category} did not have enough samples...')
        # from pudb import set_trace; set_trace()
        return 0, 0, 0, pd.DataFrame()
    if show_acc:
        print('correct', cor)
        print('total count', total)
        print(f'accuracy on {date}:', cor/total)
        print(marked_df.head(50))
    return cor, total, acc, marked_df

def accuracy_across_days(dates, selection_type='total', num_picks=6, category='all', odds_type='standard', show_acc=False):
    total_correct = 0
    total_count = 0
    acc_by_day = []
    for date in tqdm(dates):
        cor, total, acc, marked_df = daily_accuracy_check(date, selection_type,num_picks, category, odds_type, show_acc)
        if not marked_df.empty and len(marked_df) >= num_picks:
            total_correct += cor
            total_count += total
            acc_by_day.append(acc)
        else:
            print(date, 'OMMITTED')
    print('total accuracy: ', total_correct/total_count)
    print('average accuracy per day: ', np.mean(acc_by_day))
    print('med accuracy per day: ', np.median(acc_by_day))
    print(f'daily accuracies over {len(acc_by_day)} days: ', acc_by_day)
    if selection_type == 'tail':
        print('num ones:', acc_by_day.count(1.0))
    print('stev: ', np.std(acc_by_day))
    return np.mean(acc_by_day), np.std(acc_by_day)

def player_data_aggregator(player_name, start_date, end_date, odds_type, category='all', last_n=10):
    all_dfs = []
    # for date in tqdm(dates):
        # cor, total, acc, marked_df = daily_accuracy_check(date, selection_type='total', num_picks=6, category='all', odds_type='standard', show_acc=True)
    con = sqlite3.connect("Data/games.sqlite")
        # try:
    players_df = pd.read_sql_query(f"select * from \"pp_2023-24\"", con, index_col="index")
    con.close()
        # except:
        #     continue
    # pruned_df = selections_by_threshold(df, selection_type, odds_type, num_picks)
        # player_df = df[df['Player'] == player_name]
        # players_df = get_player_results(date, category='all')
    # if players_df.empty:
    #     continue
    if category != 'all':
        players_df = players_df[(players_df['Player'] == player_name) & (players_df['Category'] == category) & (players_df['Odds'] == odds_type)]
    else:
        # pd.to_datetime(
        players_df = players_df[(players_df['Player'] == player_name) & (players_df['Odds'] == odds_type)]
    players_df['Date'] = pd.to_datetime(players_df['Date'])
        # print(players_df)
    players_df = players_df[(players_df['Date'] > start_date) & (players_df['Date'] < end_date)].dropna().tail(last_n).reset_index(drop=True).drop(columns=['Category', 'Player', 'Odds'])
    players_df['Date'] = players_df['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    return players_df

def create_total_pp_df():
    con = sqlite3.connect("Data/games.sqlite")
    dates = generate_date_range('01-01-2024', '03-09-2024')
    list_of_dfs = []
    for date in tqdm(dates):
        df = get_player_results(date, category='all')
        list_of_dfs.append(df)
    total_df = pd.concat(list_of_dfs, ignore_index=True)
    total_df.to_sql(f"pp_2023-24", con, if_exists="replace")
    con.close()
    return total_df

def best_players(last_n):
    con = sqlite3.connect("Data/games.sqlite")
    players_df = pd.read_sql_query(f"select * from \"pp_2023-24\"", con, index_col="index")
    con.close()
    cats = ['Fantasy Score', 'Defensive Rebounds','Offensive Rebounds', 
'3-PT Attempted', 'Free Throws Made', 'FG Attempted', '3-PT Made', 
'Blocked Shots', 'Steals', 'Turnovers', 'Rebs+Asts', 'Blks+Stls', 'Pts+Rebs+Asts', 'Pts+Rebs', 'Pts+Asts', 'Points', 'Rebounds','Assists']
    for cat in cats:
        print(cat)
        players_cat_df = players_df[(players_df['Category'] == cat) & (players_df['Odds'] == 'standard')]
        # print(len(players_cat_df))
        players_prev_df = players_cat_df.groupby(['Player']).filter(lambda x: len(x) >= last_n)
        # print(len(players_prev_df))
        players_prev_df = players_prev_df.groupby(['Player']).tail(last_n)
        # print(len(players_prev_df))
        players_ranked = players_prev_df.groupby(['Player'])['Outcome'].value_counts(normalize=True).unstack()
        # Select only 'CORRECT' column and find players with the highest proportion
        try:
            players_cor_ranked = players_ranked['CORRECT'].sort_values(ascending=False).head()
        except:
            print('not enough samples for ', cat)
            continue
        print(players_cor_ranked)
     
# display_evaluate_pp(date='01-08-2024', category='all') '01-02-2024',
# , '01-02-2024', '01-03-2024', '01-05-2024','01-06-2024','01-07-2024', '01-08-2024', '01-09-2024'
# 
 
dates = ['01-01-2024', '01-02-2024', '01-03-2024', '01-05-2024','01-06-2024','01-07-2024', '01-08-2024', '01-09-2024', '01-10-2024', '01-11-2024', '01-12-2024','01-15-2024', '01-16-2024',
'01-17-2024', '01-18-2024', '01-20-2024', '01-22-2024', '01-23-2024', '01-24-2024', '01-25-2024', '01-26-2024', '01-27-2024', '01-28-2024',  '01-29-2024',  '01-30-2024', '01-31-2024', '02-01-2024', 
'02-02-2024','02-03-2024', '02-04-2024', '02-05-2024', '02-07-2024', '02-08-2024', '02-09-2024', '02-10-2024', '02-11-2024', '02-12-2024', '02-13-2024', '02-14-2024', '02-15-2024', '02-22-2024', '02-23-2024', '02-24-2024']
# week_dates = [['01-01-2024', '01-02-2024', '01-03-2024', '01-05-2024','01-06-2024','01-07-2024', '01-08-2024'], ['01-15-2024', '01-16-2024', '01-17-2024', '01-18-2024', '01-20-2024', '01-22-2024', '01-23-2024']]
# create df by DATE not category...
# update_df()
# total_df=None
# slxn = 'total'
# odds_type = 'standard'
# total_df = create_total_pp_dset(dates, selection_type=slxn, odds_type=odds_type)
# analyze_picks(total_df, selection_type=slxn, odds_type=odds_type)
# for date in dates:
#     load_prize_picks(date, 'all')
# daily_accuracy_check('02-23-2024', 'total', odds_type='standard', num_picks=6, show_acc=True)
# optimal_gates.keys()
# dates = ['01-26-2024', '01-27-2024', '01-28-2024',  '01-29-2024',  '01-30-2024']
# for cat in optimal_gates.keys():
#     print(cat)
# sim_acc = []
# sim_stev = []
# for x in tqdm(range(30)):
# acc, stev = accuracy_across_days(dates, selection_type='total', num_picks=6, category='all', show_acc=True)
#     sim_acc.append(acc)
#     sim_stev.append(stev)
# print('accuracy: ', sim_acc)
# print('ave mean:', np.mean(sim_acc))
# print('ave stev:', np.mean(sim_stev))
# create_total_pp_df()
# dates = ['01-17-2024']
# evaluate_prize_picks('01-24-2024')


#     try:
#         load_display_evaluate_pp(date, 'all')
#     except:
#         from pudb import set_trace; set_trace()
# display_evaluate_pp(date='01-24-2024', category='all', num_picks=6)

# accs = [0.6571428571428571, 0.59375, 0.52, 0.5641025641025641, 0.5714285714285714, 0.723404255319149, 0.5714285714285714, 0.4375, 0.4857142857142857, 0.5666666666666667, 0.5833333333333334, 0.6, 0.7758620689655172, 0.5882352941176471, 0.6071428571428571, 0.49411764705882355]
# print('stev:', np.std(accs))
# print('sample mean error: ', np.std(accs)/np.sqrt(6))

# display_prize_picks(date='01-11-2024',category='all')

# best_thresholds = aggregate_preds_across_intervals(dates)
# print(best_thresholds)

    # from pudb import set_trace; set_trace()
# profits = []
# for x in tqdm(range(30)):
#     running_profit, balance_by_day = simulate_profits(dates, 6, 5, sampling_type='random')
#     print(f'Final over {len(dates)} days: ', running_profit)
#     print(balance_by_day)
#     profits.append(running_profit)
# print(profits)
# print('average profit:', np.mean(profits))
# print('median profit:', np.median(profits))
# print('stev:', np.std(profits))
# print('max:', np.max(profits))
# print('min:', np.min(profits))
# print('1st pctile:', np.percentile(profits, 1))
# print('5th pctile:', np.percentile(profits, 5))
# print('10th pctile:', np.percentile(profits, 10))
# print('25th pctile:', np.percentile(profits, 25))
# print('75th pctile:', np.percentile(profits, 75))
# print('90th pctile:', np.percentile(profits, 90))
# print('95th pctile:', np.percentile(profits, 95))
# print('99th pctile:', np.percentile(profits, 99))
# for date in dates:
# load_display_future_data(date='03-05-2024', category='all', num_picks=6, display_only=True, sampling_type='total', odds_type='standard')
#     daily_accuracy_check(date, 'total', num_picks=6, show_acc=True)

# pairings = list(combinations(['Donovan Mitchell', 'Jarrett Allen', 'Dwight Powell', 'Shai Gilgeous-Alexander', 'Derrick Jones Jr.', 'Kyrie Irving'],2))
# print(pairings)

# load_display_evaluate_pp('01-26-2024', 'all')
# display_prize_picks('01-11-2024', 'all')
#TODO: create one big ass df...

# df = get_stev('01-28-2024')
# from pudb import set_trace; set_trace()
# # Connect to the SQLite database (replace 'your_database.db' with the actual database file)

# # for date in dates:
#     # from pudb import set_trace; set_trace()
# df = pd.read_sql_query(f"select * from \"pp_preds_{'01-22-2024'}\"", conn, index_col="index")
# print(randomize_selections_by_threshold(df, 6))
# #     from pudb import set_trace; set_trace()
# con.close()
# # # Create a cursor object
# cursor = conn.cursor()
# load_prize_picks('01-26-2024')
# # Execute a query to get the table names
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# # Fetch all the results
# tables = cursor.fetchall()

# # Print the table names
# print("Tables in the database:")
# for table in tables:
#     print(table[0])

# # Close the connection
# conn.close()




