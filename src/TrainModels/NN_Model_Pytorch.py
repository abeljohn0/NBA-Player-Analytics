import sqlite3
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import sys
import os
import shutil 
import torch
import datasets
from torch.utils.data import random_split, DataLoader, TensorDataset, SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from transformers import AdamW
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.ProcessData.Get_Data import get_player_id
from src.Utils.Dictionaries import confidence_intervals, cat_to_metric, pos_id, metric_to_cat


class LSTM_NN(nn.Module):
    def __init__(self, input_size=55, hidden_size=64, num_layers=2, dropout_rate=0.3):
        super(LSTM_NN, self).__init__()
        self.lstm = nn.LSTM(input_size, input_size, num_layers, batch_first=True, bidirectional=False, dtype=torch.float32)
        self.layer1 = nn.Linear(108, hidden_size*4)
        self.layer2 = nn.Linear(hidden_size*4, hidden_size*4)
        self.layer3 = nn.Linear(hidden_size*4, hidden_size*2)
        self.layer4 = nn.Linear(hidden_size*2, hidden_size*2)
        self.layer5 = nn.Linear(hidden_size*2, hidden_size*2)
        self.layer6 = nn.Linear(hidden_size*2, hidden_size*2)                
        self.layer7 = nn.Linear(hidden_size*2, hidden_size)
        self.layer8 = nn.Linear(hidden_size, hidden_size)
        self.layer9 = nn.Linear(hidden_size, hidden_size)
        self.layer10 = nn.Linear(hidden_size, hidden_size)
        self.layer11 = nn.Linear(hidden_size, hidden_size)
        self.layer12 = nn.Linear(hidden_size, hidden_size)
        self.layer13 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer14 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer15 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer16 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer17 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.layer18 = nn.Linear(hidden_size // 2, hidden_size // 2)        
        self.layer19 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer20 = nn.Linear(hidden_size // 4, hidden_size // 4)
        self.layer21 = nn.Linear(hidden_size // 4, hidden_size // 4)
        self.layer22 = nn.Linear(hidden_size // 4, hidden_size // 4)
        self.layer23 = nn.Linear(hidden_size // 4, hidden_size // 4)
        self.layer24 = nn.Linear(hidden_size // 4, hidden_size // 4)
        self.layer25 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.layer26 = nn.Linear(hidden_size // 8, hidden_size // 8)
        self.layer27 = nn.Linear(hidden_size // 8, hidden_size // 8)
        self.layer28 = nn.Linear(hidden_size // 8, hidden_size // 8)
        self.layer29 = nn.Linear(hidden_size // 8, hidden_size // 8)
        self.layer30 = nn.Linear(hidden_size // 8, hidden_size // 8)
        self.layer31 = nn.Linear(hidden_size // 8, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_input = x[:, :5, :]
        nn_input = x[:, 5, :54]
        h0 = torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.shape[0], self.lstm.hidden_size)

        out, _ = self.lstm(lstm_input, (h0, c0))
        out = out[:, 4, :54]
        # x = torch.cat((out, nn_input), dim=1)
        x = torch.cat((out, nn_input), dim=1)
        
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.relu(self.layer7(x))
        x = torch.relu(self.layer8(x))
        x = torch.relu(self.layer9(x))
        x = torch.relu(self.layer10(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer11(x))
        # # x = self.dropout(x)
        x = torch.relu(self.layer12(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer13(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer14(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer15(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer16(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer17(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer18(x))
        # x = self.dropout(x)        
        x = torch.relu(self.layer19(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer20(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer21(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer22(x))
        x = torch.relu(self.layer23(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer24(x))
        # x = self.dropout(x)
        x = torch.relu(self.layer25(x))
        x = torch.relu(self.layer26(x))
        x = torch.relu(self.layer27(x))
        x = torch.relu(self.layer28(x))
        x = torch.relu(self.layer29(x))
        x = torch.relu(self.layer30(x))
        x = self.dropout(x)
        x = self.layer31(x)
        # from pudb import set_trace; set_trace()
        return x


def scale_cat_drop_cols(df, target_vals=None):
    # from pudb import set_trace; set_trace()
    if target_vals is not None:
        # pathway for training+testing. Keeping DATE for sorting purposes
        df = df.drop(columns=['TEAM_ID', 'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 
                'E_TOV_PCT', 'E_USG_PCT', 'E_PACE',  
                'E_OFF_RATING_RANK', 'E_DEF_RATING_RANK', 'E_NET_RATING_RANK', 'E_TOV_PCT_RANK', 
                'E_USG_PCT_RANK', 'E_PACE_RANK', 'TEAM_E_OFF_RATING', 'TEAM_E_DEF_RATING', 
                'TEAM_E_NET_RATING', 'TEAM_E_PACE', 'MATCHUP', 'GAME_ID', 'OPP_ID', 'DATE'])
    else:
        # pathway for evaluation
        # no longer dropping playerID here so we can use for LSTM stuff...
        df = df.drop(columns=['TEAM_ID', 'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 
                'E_TOV_PCT', 'E_USG_PCT', 'E_PACE', 'E_OFF_RATING_RANK', 'E_DEF_RATING_RANK', 
                'E_NET_RATING_RANK', 'E_TOV_PCT_RANK', 'E_USG_PCT_RANK', 'E_PACE_RANK', 'TEAM_E_OFF_RATING', 
                'TEAM_E_DEF_RATING', 'TEAM_E_NET_RATING', 'TEAM_E_PACE'])
    # try:
    df.astype(float)
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(df)
    # df = pd.DataFrame(normalized_data, columns=df.columns)
    # df = df[cols]
    cols_to_keep = ['PLAYER_ID','AGE','SZN_FGM','SZN_FGA','SZN_FG_PCT','SZN_FG3M','SZN_FG3A','SZN_FG3_PCT','SZN_FTM',
        'SZN_FTA','SZN_FT_PCT','SZN_OREB','SZN_DREB','SZN_REB','SZN_AST','SZN_TOV','SZN_STL',
        'SZN_BLK','SZN_BLKA','SZN_PF','SZN_PFD','SZN_PTS','SZN_PLUS_MINUS','SZN_NBA_FANTASY_PTS',
        'SZN_DD2','SZN_TD3','OFF_RATING','sp_work_OFF_RATING','DEF_RATING',
        'NET_RATING','sp_work_NET_RATING','AST_PCT','AST_TO','AST_RATIO','OREB_PCT','DREB_PCT',
        'REB_PCT','TM_TOV_PCT','EFG_PCT','TS_PCT','USG_PCT','PACE','PACE_PER40', 'TEAM_OFF_RATING','TEAM_DEF_RATING',
        'TEAM_NET_RATING','HOME', 'DAYS_REST', 'OPP_DEF_RATING']
    # , 'POSITION'
    # if target_vals is not None:
    #     df = df[cols_to_keep[:-1] + [target_vals] + [cols_to_keep[-1]]]
    # forgot to 0-index POSITIONs. TODO Remove after next dset update
    nan_index = df[df.isna().any(axis=1)].index
    df = df.drop(nan_index)
    df = df.reset_index(drop=True)
    player_pos = df['POSITION'].copy()
    if target_vals is not None:
        targets = df[target_vals].copy()
        targets = targets.sum(axis=1)
    num_classes = 6
    df = df[cols_to_keep]
    dset_arr = torch.tensor(df.drop(columns=['PLAYER_ID']).values)
    pos_encodings = F.one_hot(torch.tensor(player_pos).long(), num_classes)
    dset_arr = torch.cat((dset_arr, pos_encodings), dim=1)
    if target_vals is not None:
        targets_tensor = torch.tensor(targets)
        dset_arr = torch.cat((dset_arr, targets_tensor.unsqueeze(1)), dim=1)
        df['TARGET'] = targets
    # torch.cat()
    # from pudb import set_trace; set_trace()
    # for x in range(len(df.max(axis=0))):
    return dset_arr, df

def create_training_tensor_lstm(target, pp_eval=False):
    season_array = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    if pp_eval:
        season_array = ["2023-24"]
    game_stats=['FGM', 'FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA',
        'FT_PCT','OREB','DREB','REB','AST','TOV','STL','BLK','BLKA','PF','PTS','PFD','PLUS_MINUS',
        'NBA_FANTASY_PTS','DD2','TD3']
    con = sqlite3.connect("../../Data/games.sqlite")
    last_n = 10
    df = pd.DataFrame(columns=['PLAYER_ID','DATE'])
    # player_lstm_df = pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_TENSOR'])
    game_stats = [item for item in game_stats if item not in target]
    # game_stats.remove(target)
    # TODO: make sure dset_arr.shape[1]...
    for season in tqdm(season_array):
        # szn_lstm_df = pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_TENSOR'])
        szn_map = {}
        df = pd.read_sql_query(f"select * from \"games_{season}\"", con, index_col="index")
        print(season, " ", df.shape[0])
        # from pudb import set_trace; set_trace()
        # target_vals = df[target]
        df = df.drop(columns=game_stats)
        # df = df.sort_values(by='DATE')
        # , 'PLAYER_ID'
        df = df.sort_values(by=['PLAYER_ID','DATE'])
        # df = df.sort_values(by=['DATE', 'PLAYER_ID'], ascending=[True, False])
        df = df.reset_index(drop=True)
        # print(df.head())
        # from pudb import set_trace; set_trace()
        dset_arr, updated_df = scale_cat_drop_cols(df, target_vals=target)
        if pp_eval:
            return dset_arr, updated_df
        for player in tqdm(updated_df['PLAYER_ID'].unique()):
            games_by_player = torch.empty(0)
            player_games = updated_df[updated_df['PLAYER_ID'] == player]
            if len(player_games) > last_n:
                first_index = player_games.index[0]
                last_index = player_games.index[-1]
                # print(last_index - first_index)
                for game_index in range(first_index+last_n, last_index+1):
                        try:
                            games_by_player = torch.cat((games_by_player, dset_arr[game_index-last_n:game_index+1].unsqueeze(0)), dim=0)
                        except Exception:
                            from pudb import set_trace; set_trace()
                szn_map[player] = games_by_player
        szn_lstm_df = pd.DataFrame(szn_map.items(), columns=['PLAYER_ID', 'PLAYER_TENSOR'])
        for x in szn_lstm_df['PLAYER_TENSOR']:
            try:
                if torch.max(x) > 10**5:
                    print(torch.max(x))
            except Exception:
                from pudb import set_trace; set_trace()
        # x = szn_lstm_df['PLAYER_TENSOR'].apply(lambda x: pickle.loads(x))
        # if season == "2023-24":
        #     from pudb import set_trace; set_trace()
        szn_lstm_df['PLAYER_TENSOR'] = szn_lstm_df['PLAYER_TENSOR'].apply(lambda x: pickle.dumps(x))
        szn_lstm_df.to_sql(f"games_{season}_lstm", con, if_exists="replace")
    # player_lstm_df.to_sql(f"games_2014-2024_lstm", con, if_exists="replace")
    con.close()

def concat_tensors(x, y):
    x = x if pd.notna(x) else torch.empty(0)
    y = y if pd.notna(y) else torch.empty(0)
    
    return torch.cat([x, y], dim=0)

def create_total_train_tensor():
    season_array = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
    con = sqlite3.connect("../../Data/games.sqlite")
    total_df = pd.DataFrame
    for season in tqdm(season_array):
        szn_lstm_df = pd.read_sql_query(f"select * from \"games_{season}_lstm\"", con, index_col="index")
        # x = pickle.loads(szn_lstm_df.iloc[0]['PLAYER_TENSOR'])
        szn_lstm_df['PLAYER_TENSOR'] = szn_lstm_df['PLAYER_TENSOR'].apply(lambda x: pickle.loads(x))
        if total_df.empty:
            total_df = szn_lstm_df
        else:
            try:
                # from pudb import set_trace; set_trace()
                total_df = pd.merge(total_df, szn_lstm_df, on='PLAYER_ID', how='outer')
                total_df = total_df.assign(concatenated_tensors=lambda df: df.apply(lambda row: concat_tensors(row['PLAYER_TENSOR_x'], row['PLAYER_TENSOR_y']), axis=1))
                # total_df['PLAYER_TENSOR_x'] = total_df['PLAYER_TENSOR_x'].fillna(total_df['PLAYER_TENSOR_y'])

                total_df = total_df.drop(columns=['PLAYER_TENSOR_y', 'PLAYER_TENSOR_x'])    
                total_df.rename(columns={'concatenated_tensors': 'PLAYER_TENSOR'}, inplace=True)
            except Exception:
                print("fuck")
    # from pudb import set_trace; set_trace()
    total_df['PLAYER_TENSOR'] = total_df['PLAYER_TENSOR'].apply(lambda x: pickle.dumps(x))
    total_df.to_sql(f"games_2014-2023_lstm", con, if_exists="replace")
    con.close()


def lstm_preprocessing(target):
    # df gets passed in here
    con = sqlite3.connect("../../Data/games.sqlite")
    create_training_tensor_lstm(target)
    create_total_train_tensor()
    
    # from pudb import set_trace; set_trace()
    cur_szn_df = pd.read_sql_query(f"select * from \"games_2023-24_lstm\"", con, index_col="index")
    total_tensor_df = pd.read_sql_query(f"select * from \"games_2014-2023_lstm\"", con, index_col="index")
    con.close()
    total_tensor_df['PLAYER_TENSOR'] = total_tensor_df['PLAYER_TENSOR'].apply(lambda x: pickle.loads(x))
    cur_szn_df['PLAYER_TENSOR'] = cur_szn_df['PLAYER_TENSOR'].apply(lambda x: pickle.loads(x))
    # nan_index = total_tensor_df[total_tensor_df['PLAYER_TENSOR'].isna().any(axis=1)].index
    # df = df.drop(nan_index)
    all_szns_playergames = list(total_tensor_df['PLAYER_TENSOR'])
    cur_szn_playergames = list(cur_szn_df['PLAYER_TENSOR'])

    data_tensor = torch.cat(all_szns_playergames, dim=0)
    cur_tensor = torch.cat(cur_szn_playergames, dim=0)

    # from pudb import set_trace; set_trace()
    dataset_size = data_tensor.shape[0]
    cur_dset_size = cur_tensor.shape[0]
    train_dataset = TensorDataset(data_tensor)
    train_indices = list(range(len(train_dataset)))
    train_indices = torch.randperm(len(train_indices))
    train_sampler = SubsetRandomSampler(train_indices)

    split_ratio = [0.8, 0.2]
    cur_dataset = TensorDataset(cur_tensor)
    val_size = int(cur_dset_size * split_ratio[0])
    test_size = int(cur_dset_size * split_ratio[1])
    indices = list(range(len(cur_dataset)))
    indices = torch.randperm(len(indices))
    val_indices = indices[:val_size]
    val_sampler = SubsetRandomSampler(val_indices)

    test_indices = indices[val_size:]
    test_sampler = SubsetRandomSampler(test_indices)


    # middle_dataset = TensorDataset(data_tensor[train_size:,:,:])
    # val_dataset, test_dataset = random_split(middle_dataset, [val_size, test_size])

    train_batch_size = 256
    val_batch_size = 72
    test_batch_size = 36
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,  sampler=train_sampler)
    val_dataloader = DataLoader(cur_dataset, batch_size=val_batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(cur_dataset, batch_size=test_batch_size, sampler=test_sampler)
    return (train_dataloader, val_dataloader, test_dataloader)

def train_lstm(model, dataloaders, num_epochs, dirpath, category):
    train_dataloader, val_dataloader = dataloaders
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.00001)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float("inf")
    progress_bar = tqdm(range(num_training_steps))

    train_loss_history = []
    val_loss_history = []
    alpha, beta, gamma = 1, 1, 1
    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss = 0
        # from pudb import set_trace; set_trace()
        for sample in train_dataloader:
            target = sample[0][:, 5, 54].to(torch.float32)
            out = model(sample[0].to(torch.float32)).squeeze(1)
          
            optimizer.zero_grad()
          # flag = sample1['labels'] == sample2['labels']; flag = flag.cpu().item();
          
            loss_val = gamma*F.mse_loss(out, target)
            # from pudb import set_trace; set_trace()
            if torch.isinf(loss_val):
                from pudb import set_trace; set_trace()
            loss_val.backward(retain_graph=True)         # compute grads
            optimizer.step()            # update weights

          # x2_classification_val = beta*loss_func_classification(update_x2);
          # x2_classification_val.backward()         # compute grads
          # optimizer.step()  

            train_loss += loss_val.item()
          # + x2_classification_val.item()
    
            lr_scheduler.step()
            progress_bar.update(1)

        avg_train_loss = train_loss/len(train_dataloader)
        print(avg_train_loss)
        # from pudb import set_trace; set_trace()

        # validation
        model.eval()
        val_loss = 0
        for sample in val_dataloader:
            target = sample[0][:, 5, 54].to(torch.float32)
            with torch.no_grad():
                out = model(sample[0].to(torch.float32)).squeeze(1)
                loss_val = gamma*F.mse_loss(out, target)
                val_loss +=  loss_val.item()
          # + x2_classification_val.item()

        avg_val_loss = val_loss / len(val_dataloader)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1}" + " | " + f"Training loss: {avg_train_loss}" + " | " + f"Validation loss: {avg_val_loss}")
        if avg_val_loss < best_val_loss:
            print("Saving checkpoint!")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                },
                f"{dirpath}/best_{cat_to_metric[category]}_lstm.pt"
            )
    return model, train_loss_history, val_loss_history  


def test(model, test_dataloader, dirpath, threshold, category, model_type='lstm+NN'):
    print('loading the best model')
    if model_type == 'lstm+NN':
        checkpoint_path = f"{dirpath}/best_{cat_to_metric[category]}_lstm.pt"
    checkpoint = torch.load(checkpoint_path)

    loaded_model_state_dict = checkpoint['model_state_dict']

    model.load_state_dict(loaded_model_state_dict)
    test_loss = 0
    pct_count = 0
    pt_cnt = 0
    pt_rando = 0
    ave_pct_error = 0
    count = 0
    for sample in test_dataloader:
        model.eval()
        with torch.no_grad():
            if model_type == 'lstm+NN':
                target = sample[0][:, 5, 54].to(torch.float32)
                out = model(sample[0].to(torch.float32)).squeeze(1)
            loss_test = F.mse_loss(out, target)
            test_loss +=  loss_test.item()
            # from pudb import set_trace; set_trace()
            for x in range(len(out)):
                result = out[x]
                rando = torch.randint(0, 15, (1, 1)).float()
                pct_error = torch.mean(torch.abs((result - target[x]) / target[x]))
                pts_off = torch.abs(result - target[x])
                # print('predicted: ', result)
                # print('actual: ', target[x])
                # print("pct error: ", pct_error*100)
                # print("pts off: ", pts_off)
                # print(" ")
                if torch.isinf(pct_error):
                    continue
                ave_pct_error += pct_error
                if pct_error <= 0.15:
                    pct_count += 1
                if pts_off < threshold:
                    # print('less than threshold!!!')
                    pt_cnt += 1
                if torch.abs(rando - target[x]) < threshold:
                    pt_rando += 1
                count += 1
    avg_test_loss = test_loss / count
    ave_pct_error = ave_pct_error / count
    # print('overall test loss: ', avg_test_loss)
    # print('w/in pct threshold: ', pct_count / len(test_dataloader.dataset))
    # print('ave pct error: ', ave_pct_error)
    confidence = pt_cnt / count
    print('num samples: ', count)
    print('w/in point threshold: ', confidence)
    print('random guessing - point threshold: ', pt_rando / count)
    # print('dset size: ', len(test_dataloader.dataset))
    return confidence
# def prizepicks(model, dec_29):

def eval(model, test_sample, dirpath, category, future=False):
    # print('loading the best model')
    checkpoint_path = f"{dirpath}/best_{category}_lstm.pt"
    checkpoint = torch.load(checkpoint_path)
    loaded_model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(loaded_model_state_dict)
    dset_arr, updated_df = create_training_tensor_lstm(category, pp_eval=True)
    # from pudb import set_trace; set_trace()
    days_prior = test_sample['GAMES_FROM_PRESENT'].values[0]
    if days_prior == 0:
            if future:
                last_five = updated_df[updated_df['PLAYER_ID'] == test_sample['PLAYER_ID'].values[0]].tail().index
            else:
                last_five = updated_df[updated_df['PLAYER_ID'] == test_sample['PLAYER_ID'].values[0]].tail(6).index[:-1]
    else:
        last_five = updated_df[updated_df['PLAYER_ID'] == test_sample['PLAYER_ID'].values[0]].tail(5+days_prior).index[:-days_prior]
    test_sample = test_sample.drop(columns=['GAMES_FROM_PRESENT'])
    context = dset_arr[last_five, :]
    if len(context) < 5:
        print('Player does not have enough starts')
        return torch.tensor(-1)
    # torch.save(model.state_dict(), 'best_model.pth')
    # print('weights saved! Comment me out now...')
    test_sample, _ = scale_cat_drop_cols(test_sample)
    zero_col = torch.zeros(test_sample.size(0), 1)
    sample = torch.cat((context, torch.cat((test_sample, zero_col), dim=1)), dim=0)
    model.eval()
    with torch.no_grad():
        out = model(sample.unsqueeze(0).to(torch.float32)).squeeze(0)
        return out[0]


def run_eval(test_sample, score, category, future=False):
    # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    prob_sum = 0
    pt_sum = 0
    if test_sample.empty:
        return -1, -1, -1
    categories = [cat_to_metric[category]]
    for cat in categories:
        model_path = dirpath + f"/best_{cat}_lstm.pt"
        if not os.path.exists(model_path):
            print('model does not exist yet!!!!')
            break
            # train_and_test_lstm(cat)
        # from pudb import set_trace; set_trace()
        result = eval(model, test_sample, dirpath, cat, future=future).item()
        pt_sum += result
        if result < 0:
            return -1, -1, -1
        if len(cat) > 1:
            cnf_intvl = confidence_intervals[metric_to_cat[tuple(cat)]]
        else:
            cnf_intvl = confidence_intervals[cat[0]]
        try:
            threshold = abs(round(result*2)/2 - score)
            prob = cnf_intvl[abs(round(result*2)/2 - score)]
        except:
            prob = 1
        prob_sum += prob
    print('predicted: ', pt_sum)
    print('line: ', score)
    print('pct confidence: ', prob_sum/len(categories))
    return prob_sum/len(categories), pt_sum, threshold


def train_and_test_lstm(category):
    # from pudb import set_trace; set_trace()
    train_dataloader, val_dataloader, test_dataloader = lstm_preprocessing(cat_to_metric[category])
    num_epochs = 30
    train_lstm(model, (train_dataloader, val_dataloader), num_epochs, dirpath, category)
    thresholds = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30]
    # thresholds = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    conf_per_gate = {}
    for x in thresholds:
        confidence = test(model, test_dataloader, dirpath, x, category, 'lstm+NN')
        conf_per_gate[x] = confidence
    # TODO: make it so that this is connected to SQL DB and I don't manually update confidence
    print(conf_per_gate)

dirpath = '/Users/abeljohn/Developer/Player-Sports-Analytics/src/TrainModels/models'
model = LSTM_NN()


# con = sqlite3.connect("../../Data/games.sqlite")
# for season in tqdm(season_array):
#     szn_df = pd.read_sql_query(f"select * from \"games_{season}\"", con, index_col="index")
#     szn_df = szn_df.drop(columns=['POSITION_x', 'POSITION_y'])
#     # from pudb import set_trace; set_trace()
#     # szn_df = szn_df.sort_values(by='DATE')
#     # szn_df = szn_df.sort_values(by='PLAYER_ID')
#     # szn_df = szn_df.reset_index(drop=True)
#     # lstm_preprocessing(szn_df, cat_to_metric[category])
#     # from pudb import set_trace; set_trace()
#     szn_df.to_sql(f"games_{season}", con, if_exists="replace")
# con.close() 
# cats = ['Points', 'Rebounds','Assists', 'Fantasy Score', 'Defensive Rebounds', 
# 'Offensive Rebounds', '3-PT Attempted', 'Free Throws Made','FG Attempted', '3-PT Made', 
# 'Blocked Shots', 'Steals', 'Turnovers', 'Pts+Rebs+Asts', 'Pts+Rebs', 'Pts+Asts', 'Rebs+Asts', 'Blks+Stls']
# for cat in cats:
#     train_and_test_lstm(cat)
#     print(cat)
# train_and_test_lstm('Fantasy Score')
# train_and_test('Points')
# train(model, (train_dataloader, val_dataloader), num_epochs, dirpath)
# prizepicks(model, dec_29)




