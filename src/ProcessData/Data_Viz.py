import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
from Get_Data import get_player_id
#  'Pts+Rebs+Asts': ['PTS','REB','AST'], 'Rebs+Asts': ['REB','AST'],'Pts+Asts': ['PTS','AST'], 'Blks+Stls': ['BLK','STL'],'Pts+Rebs': ['PTS','REB'],  

cats = ['PTS', 'REB', 'AST', 
'NBA_FANTASY_PTS', 'DREB', 'OREB', 
'FG3A', 'FG3M', 'FTM', 'FGA', 
'BLK', 'STL', 'TOV']

con = sqlite3.connect("../../Data/games.sqlite")
szn_df = pd.read_sql_query(f"select * from \"games_2023-24\"", con, index_col="index")
# player = "Stephen Curry"
# player_id = get_player_id(player)
# from pudb import set_trace; set_trace()
# # metrics = szn_df[szn_df['PLAYER_ID'] == player_id][cats]
metrics = szn_df.groupby(['POSITION', 'DATE'])['PTS'].mean().reset_index()
# fig = ff.create_distplot(np.transpose(metrics.to_numpy()), group_labels=['PTS'], curve_type='kde')
# fig.update_layout(title_text=f'Distribution of Stats for: {player}. Sample Size: {len(metrics)}')
# fig.show()
# from pudb import set_trace; set_trace()
sns.set(style="whitegrid")
g = sns.FacetGrid(metrics.tail(), col='DATE', row='POSITION', margin_titles=True)
g.map(sns.kdeplot, "PTS", color='steelblue')

# Set labels and titles
g.set_axis_labels("Points per Game", "Density")
g.set_titles(col_template="uh", row_template="idk")
# Adjust the layout for better spacing
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Kernel Density Estimate of Points per Game by Position Over Time')
plt.savefig('/Users/abeljohn/Developer/NBA-Machine-Learning-Sports-Betting/src/ProcessData/plot.png')
print('plot saved')
# Show the plot
plt.show()