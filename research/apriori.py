import pandas as pd
import networkx as nt
import numpy as np
import apyori as ap
from mlxtend.frequent_patterns import apriori, association_rules
from gsppy.gsp import GSP


def is_action_exist_in_pattern(actions_list, pattern):
    return tuple([i in set(pattern) for i in actions_list])

# data = pd.read_csv("for_apriori_algo_2003.csv")
data = pd.read_csv("csv_files/for_apriori_algo_2003.csv")
# player = pd.read_csv("csv_files/player_info_2003.csv")

# avoid na
data = data.fillna('NA')

# dictionary of paterns
chunck_ind = data[data.ind_canasta_2 == 1].index
data_lst = []
for i, j in zip(chunck_ind[0:-1], chunck_ind[1::]):
    temp_dict = {'players': data.loc[i+1:j+1, 'player_name'].to_list()
        , 'teams': data.loc[i+1:j+1, 'team'].to_list()
        ,'play_number': data.loc[i+1:j+1, 'play_number'].to_list()
        , 'pattern': data.loc[i+1:j+1, 'action'].to_list()
        , 'shot_player': data.loc[j, 'player_name']
        , 'shot_team': data.loc[j, 'team']
        , 'n': len(data.loc[i+1:j+1, 'action'].to_list())
                 }
    data_lst.append(temp_dict)

# Creating list of list for the apriori algo
app_list = [i['pattern'] for i in data_lst]

# appriori algo

# Using the second package for apriori
unique_action = data['action'].unique()
temp_map = map(lambda x: is_action_exist_in_pattern(unique_action, x), app_list)
temp_to_df = []
for i in temp_map:
    temp_to_df.append(i)
data_ap = pd.DataFrame(data = temp_to_df, columns=unique_action)
# data_ap = data_ap.iloc[0:10000,:]
frequent_itemsets = apriori(data_ap, min_support=0.05, use_colnames=True, max_len=7, min_len = 1)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# printing the frequent itemset
top_5 = frequent_itemsets.groupby('length')['support'].nlargest(4).reset_index().loc[:,'level_1']
final_top = frequent_itemsets.iloc[top_5.values,:]




