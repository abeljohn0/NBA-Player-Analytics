import argparse
from src.ProcessData.Prize_Picks import load_display_future_data, daily_accuracy_check, accuracy_across_days, player_data_aggregator, generate_date_range, create_total_pp_df, best_players
from datetime import datetime, timedelta
import re

def main():
    parser = argparse.ArgumentParser(description='this is a program to analyze and predict player scoring outputs')

    # cmd_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('command', help='''available commands:\n
    -todays_picks: Get today's top rated picks\n
    -eval: Get the results for picks made on a prior date\n
    -player: Get metrics for a specific player''')
    # cmd_group.add_argument('eval_date', help='Get the results for picks made on a prior date')
    parser.add_argument('-display_only', action='store_true', help='reload picks')

    pick_group = parser.add_mutually_exclusive_group(required=False)
    pick_group.add_argument('-demon', action='store_true', help='include demon')
    pick_group.add_argument('-goblin', action='store_true', help='include goblin')

    parser.add_argument('-date', type=str, help='which date to evaluate')
    parser.add_argument('-date_range', type=str, help='range of dates - format: \"XX-XX-XXXX -> YY-YY-YYYY\"')

    parser.add_argument('-player_name', type=str, help='player you would like to learn more about')
    parser.add_argument('-last_n', type=int, default=5, help='how many games of the player would you like to view?')
    parser.add_argument('-category', type=str, default='all', help='which stat would you like to see?')

    args = parser.parse_args()

    odds_type='standard'
    if args.goblin:
        odds_type = 'goblin'
    if args.demon:
        odds_type = 'demon'
    if args.command == 'todays_picks':
        date=datetime.now().strftime('%m-%d-%Y')
        if args.date:
            date=args.date
        print(date)
        load_display_future_data(date=date, category='all', num_picks=6, display_only=args.display_only, sampling_type='total', odds_type=odds_type)
    if args.command == 'eval':
        if args.date:
            pattern = re.compile(r'^\d{2}-\d{2}-\d{4}$')
            match = pattern.match(args.date)
            if not match:
                print('date not entered correctly! Please enter in the following format: XX-XX-XXXX')
            else:
                daily_accuracy_check(args.date, 'total', odds_type=odds_type, num_picks=6, show_acc=True)
        if args.date_range:
            print(args.date_range)
            pattern = re.compile(r'(\d{2}-\d{2}-\d{4})')
            matches = pattern.findall(args.date_range)
            if len(matches) != 2:
                print('date not entered correctly! Please enter in the following format:\"XX-XX-XXXX -> YY-YY-YYYY\"')
            # start_date, end_date = pattern.findall(matches)
            dates = generate_date_range(matches[0], matches[1])
            acc, stev = accuracy_across_days(dates, selection_type='total', num_picks=6, category='all', odds_type=odds_type, show_acc=False)
            print(f'accuracy across {len(dates)} days:', acc)
            print(f'stev across {len(dates)} days:', stev)
                # daily_accuracy_check(args.date, 'total', odds_type='standard', num_picks=6, show_acc=True)
    if args.command == 'player':
        end_date = datetime.now().strftime('%m-%d-%Y')
        start_date = (datetime.now() - timedelta(days=2*args.last_n)).strftime('%m-%d-%Y')
        dates = generate_date_range(start_date, end_date)
        print(dates)
        player_data_aggregator(args.player_name, dates, category=args.category, odds_type=odds_type)

def meme():
    create_total_pp_df()

def get_stats(last_n):
    best_players(last_n)

if __name__ == '__main__':
    main()
    # meme()
    # get_stats(4)