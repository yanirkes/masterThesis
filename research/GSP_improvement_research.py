
import pandas as pd
import utils.postgres_config as psql
import time
from matplotlib import pyplot as plt

def is_action_exist_in_pattern(actions_list, pattern):
    return tuple([i in set(pattern) for i in actions_list])


def reutrn_dist_of_k(k, data, thres):
    return pd.Series([ data[str(k)][i] for i in data[str(k)] if  data[str(k)][i]  > thres ])

def reutrn_freq_k_sequence(k, data, thres):
    return pd.Series([ i for i in data[str(k)] if  data[str(k)][i]  > thres ])

def reutrn_freq_k_sequence_with_freq(k, data, thres):
    return pd.DataFrame([ (i,data[str(k)][i])  for i in data[str(k)] if  data[str(k)][i]  > thres ])


con = psql.PostgressCon()
con.connect()
df = con.execute_query_with_headers("""select  game_id, action, team, play_number, player_name, 
                                       case when action in ('Canasta de 3', 'Canasta de 2','Canasta de 1','FIN DE PERIODO') then 1 else 0 end ind_canasta_2
                                       from stg.all_data_info
                                       where year >= 2003
                                       and action  not in ('MARCA DE MINUTO','Sale a Banquillo','Entra a Pista')
                                        order by 1,4 asc
                                        --and action  in ('Canasta de 3','Asistencia','Pérdida', 'Intento fallado de 2','Canasta de 2')"""

                                    )
df = pd.DataFrame(data = df[0], columns=df[1])

df = df.fillna('NA')

# dictionary of paterns
chunck_ind = df[df.ind_canasta_2 == 1].index
chunck_ind = chunck_ind.insert(0, 0)
df_lst = []

for i, j in zip(chunck_ind[0:-1], chunck_ind[1::]):
    j = i if j == 6 else j
    i = -1 if i == 3 else i
    temp_dict = {'players': df.loc[i + 1:j, 'player_name'].to_list()
        , 'teams': df.loc[i + 1:j, 'team'].to_list()
        , 'play_number': df.loc[i + 1:j, 'play_number'].to_list()
        , 'pattern': df.loc[i + 1:j, 'action'].to_list()
        , 'shot_player': df.loc[j, 'player_name']
        , 'shot_team': df.loc[j, 'team']
        , 'n': len(df.loc[i + 1:j, 'action'].to_list())
                 }
    df_lst.append(temp_dict)

app_list = [i['pattern'] for i in df_lst if len(i['pattern']) <= 30]

t0 = time.time()
dist_dict = {}
for  x in range(1,31):
    dist_dict.setdefault(str(x),{})
    temp = [ i for i in  app_list if len(i) == x]
    for j in temp:
        j = ','.join(j)
        dist_dict[str(x)][str(j)] = dist_dict.get(str(x),{}).get(j,0)+1

sequence_size_dict = {x: len([i for i in app_list if len(i) == x]) for x in range(0,31) }

plt.bar(sequence_size_dict.keys(), sequence_size_dict.values())

t1 = time.time()
print("DONE PREPROCESSING IN ", round((t1 - t0)), " SEC")
reutrn_dist_of_k(2,dist_dict, 1).hist(bins = 30)

result = reutrn_freq_k_sequence(7,dist_dict, 40)

freuqent_df = reutrn_freq_k_sequence_with_freq(7,dist_dict, 30)
freuqent_df.columns = ['sequence', 'counts']
freuqent_df.sort_values(by= 'counts', inplace=True, ascending=False)

# Creating list of list for the apriori algo
