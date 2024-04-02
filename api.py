import os
import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# print(os.path.dirname(__file__))
# sys.path.append(project_root)
from flask import Flask, request
from src.ProcessData.Prize_Picks import load_display_future_data, daily_accuracy_check, accuracy_across_days, player_data_aggregator, generate_date_range
from src.ProcessData.Injury_Report import get_injury_data
from datetime import datetime, timedelta
import json
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)


@app.before_request
def before_request():
    if request.method == "OPTIONS":
        return "", 200

@app.route('/player', methods=['POST'])
def index():
    content = request.json
    player_name = content['player_name']
    category = content['category']
    print(player_name)
    print(category)
    end_date = datetime.now().strftime('%m-%d-%Y')
    start_date = '01-01-2024'
    # dates = generate_date_range(start_date, end_date)
    df = player_data_aggregator(player_name, start_date, end_date, category=category, odds_type='standard')
    json_data = df.to_json(orient='columns')
    return json_data

@app.route('/picks', methods=['GET'])
def today_picks():
    date = (datetime.now()).strftime('%m-%d-%Y')
    print(date)
    try:
        today_df = load_display_future_data(date=date, category='all', num_picks=6, display_only=True, sampling_type='total', odds_type='standard').drop(columns=['Odds', 'Prob'])
        json_data = today_df.to_json(orient='columns')
    except:
        json_data = {'no data': 'no data'}
    return json_data

@app.route('/injuries', methods=['POST'])
def today_injuries():
    content = request.json
    # print(request.json)
    team_name = content['team']
    # print('henlo')
    # print(content)
    today_df = get_injury_data(team_name).drop(columns=['Team'])
    print(today_df)
    json_data = today_df.to_json(orient='columns')
    return json_data

@app.route('/player_photo', methods=['POST'])
def get_player_photo():
    content = request.json
    # print(request.json)
    player_name = content['player_name']
    with open('Data/player_photos.json', 'r') as json_file:
    # Load the JSON data
        player_pics = json.load(json_file)
    first_and_last = player_name.split(" ")
    formatted_input = first_and_last[1][:5] + first_and_last[0][:2] + '01'
    url = player_pics[formatted_input.lower()]
    return {'player_photo': url}

if __name__ == '__main__':
    app.run(debug=True)