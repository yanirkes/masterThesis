import pandas as pd
import threading as thrd
import time
import numpy as np
import datetime
import queue
import utils.constant_class as const
import sys

def change_player_values(df, name_to_filer, new_name, by_first_name = False):
    if by_first_name:
        mask = (df["first_name"] == name_to_filer)&(df["exist"] == False)
    else:
        mask = (df["player_name"] == name_to_filer)&(df["exist"] == False)
    df.loc[mask, 'partner'] = new_name
    df.loc[mask, 'exist'] = 'True'
    return df

def alter_player_name(name, player_info, db_df):
    mask = db_df["player_name"] == name
    new_name = player_info[player_info['player_name'] == name]['partner'][0]
    db_df.loc[mask, 'player_name'] = new_name


def fill_missing_year_temp(name, data):
    player_info_df = data[data["player"] == name]
    if not player_info_df['current_age'].isnull().all():
        current_age = list(player_info_df[~ player_info_df['current_age'].isnull()].loc[:,"current_age"])[0]
        mask = (data['current_age'].isnull())&(data["player"] == name)
        data.loc[mask, 'current_age'] = current_age
    else:
        print("all null")

def alter_team_name(new_team_name, team_list, data):
    mask = data["team_name"].isin(team_list)
    data.loc[mask, 'team_name'] = new_team_name

def numerize_baketball_shots(type_of_shot):
    if not type_of_shot in []:
        raise Exception("The type of shoot is invalid, should be tira 1, 2 or 3 or Mate")
    if type_of_shot == 'Canasta de 1':
        return 1
    elif type_of_shot == 'Canasta de 2':
        return 2
    elif type_of_shot == 'Mate':
        return 2
    else :
        return 3

def get_timestamp(ind, play_time_df, df, fun, hour_lst):
        play_number = play_time_df[ind ]
        if fun == "max":
            hour_lst.append(df[(df["year"] == ind[0] )&(df["game_id"] == ind[1])&(df["play_number"] == play_number-1)].loc[:,"timeStamp"])
        else:
            hour_lst(df[(df["year"] == ind[0] )&(df["game_id"] == ind[1])&(df["play_number"] == play_number)].loc[:,"timeStamp"].values())


data = pd.read_csv('C:/Users/yanir/Documents/UC3M/Thesis/full_db.csv')
result = data.iloc[:,1::]
result.loc[:, 'Denominacion_Larga'] = result.loc[:, 'Denominacion_Larga'].fillna("")
result.loc[:, 'Apodo_Largo'] = result.loc[:, 'Apodo_Largo'].fillna("")
result.loc[:, 'Descripcion'] = result.loc[:, 'Descripcion'].fillna("")
result.loc[:, 'eq'] = result.loc[:, 'eq'].fillna("")
result.loc[:, 'Cod_Jugador'] = result.loc[:, 'Cod_Jugador'].fillna("")
result.loc[:, 'coordX'] = result.loc[:, 'coordX'].fillna(10000)
result.loc[:, 'coordY'] = result.loc[:, 'coordY'].fillna(10000)
result.loc[:, 'TiempoMarcador'] = result.loc[:, 'TiempoMarcador'].fillna(-1)


ls = ["<e9>",
"<e1>",
"<f1>",
"<ed>",
"<fa>",
"<c1>",
"<f3>",
"<da>",
"<e0>",
"<eb>",
"<fc>",
"<e7>",
"<d3>"]

# Find substrings "<>" in the df to mapp the dict
for i in ls:
    print(i)
    mask = result[['Descripcion','Denominacion_Larga','Apodo_Largo']].str.contains(i)
    temp = result[mask]

dic = {
    "<e9>": "é",
"<e1>": "á",
"<f1>": "ñ",
"<ed>": "í",
"<fa>": "ú",
"<c1>": "Á",
"<f3>": "ó",
"<da>": "Ú",
"<e0>": "à",
"<eb>": "ë",
"<fc>": "ü",
"<e7>": "ç",
"<d3>": "Ó"
}

# Replace letter
for i in ls:
    result[['Descripcion','Denominacion_Larga','Apodo_Largo']] = result[['Descripcion','Denominacion_Larga','Apodo_Largo']].replace({i: dic[i]}, regex=True)

result.loc[:, "Apodo_Largo"] = result.loc[:, "Apodo_Largo"].replace({',':""}, regex = True)

# From object to strin
#  "Denominacion_Larga","Descripcion","Apodo_Largo"
object_lst_to_conver = ["ed",
                        "eq",
                        "Cod_Jugador",
                        "Dorsal","Denominacion_Larga","Descripcion","Apodo_Largo"]

for i in object_lst_to_conver:
    print(i)
    result[i] = result[i].astype('string')

# Cordx Cordy nan => 0
result.loc[:, 'Dorsal'] = result.loc[:, 'Dorsal'].fillna("101")
result.loc[:, 'Dorsal'] = result.loc[:, 'Dorsal'].replace("EQ","100")
result['coordX'] = result['coordX'].astype(int)
result['coordY'] = result['coordY'].astype(int)
result['TimeStamp'] = pd.to_datetime(result['TimeStamp'])

# Replace year format, e.g from 2003-2004 = > 2003
result = result.copy()
for i in range(2003,2014):
    rep = str(i)+'-'+str(i+1)
    result[result['ed'] == rep] = result[result['ed'] == rep].replace(rep, str(i))

result['ed'] = result['ed'].astype(int)

# change full_db columns' name
result.columns = ['year', 'game_id', 'play_number', 'action_code', 'action', 'team','team_name', 'player_code', 'player_name', 'player_shirt_number', 'coordx', 'coordy', 'time_marker', 'timeStamp']

# compare player df and full_db df
player = pd.read_csv("player_detail.csv")
full_db = pd.read_csv("player_detail_4.csv")
player = pd.concat([player,full_db])

player.to_csv("player_detail.csv", index=False)


player_list = list(player["player"].unique())
player_res_list = list(result["player_name"].unique())
player_res_list.remove("")
temp = player_list
compare_name_df = pd.DataFrame(columns = ["player_name","partner", "exist"])
for player_ in player_res_list:
    isin_ = False
    for partner in player_list:
        if set(player_.lower().split(" ")) == set(partner.lower().split(" ")):
            isin_ = True
            temp_df = pd.DataFrame({"player_name":[player_],"partner":[partner], "exist":[isin_]})
            compare_name_df = pd.concat([compare_name_df,temp_df])
            player_list.remove(partner)
    if not isin_:
            temp_df = pd.DataFrame({"player_name": [player_], "partner": [""], "exist": [isin_]})
            compare_name_df = pd.concat([compare_name_df,temp_df])
compare_name_df['first_name'] = compare_name_df["player_name"].str.split(" ",1,expand=True).loc[:,0]

temp_name_missing = sorted(list(compare_name_df[compare_name_df["exist"] == False].loc[:,'player_name']))
temp_name_missing = [i.split(' ') for i in temp_name_missing]
temp_name_missing.sort(key = lambda x: x[0] if len(x)== 1 else x[1])
player_list = sorted(player_list)
temp = [i.lower().split(' ') for i in player_list]

name_missing = {x[0]: [player_list[ind] for ind, val in enumerate(temp) if x[0].lower() in val] for x in temp_name_missing }
one_name_lst = []
two_name_lst = []
multi_name_lst = []
for i in name_missing.keys():
    if len(name_missing[i]) == 1:
        one_name_lst.append(i)
    elif len(name_missing[i]) > 1:
        multi_name_lst.append(name_missing[i])
        print(name_missing[i])

for first_name in one_name_lst:
    if not compare_name_df[compare_name_df["first_name"] == first_name].empty:
        name = ''.join(name_missing[first_name][0])
        compare_name_df = change_player_values(compare_name_df, first_name, name, True)

# Hernández
# ['Antonio Hernández', 'José Manuel Hernández', 'Raúl Hernández']
# compare_name_df[(compare_name_df["first_name"] == 'Hernández')&((compare_name_df["exist"] == False))]
compare_name_df = change_player_values(compare_name_df, 'Hernández A.', 'Antonio Hernández' )

# López
# [ ]
compare_name_df = change_player_values(compare_name_df, 'López Vinuesa', 'Rai López de Vinuesa' )
compare_name_df = change_player_values(compare_name_df, 'López Valera José', 'José López' )
compare_name_df = change_player_values(compare_name_df, 'López', 'López Isaac' )
compare_name_df = change_player_values(compare_name_df, 'López Juanjo', 'Juan José López' )
compare_name_df = change_player_values(compare_name_df, 'López Alejandro', 'Álex López' )

# Santos
compare_name_df = change_player_values(compare_name_df, 'de los Santos N.', 'Nicolas de los Santos' )
# García
compare_name_df = change_player_values(compare_name_df, 'de los Santos N.', 'Nicolas de los Santos' )
compare_name_df = change_player_values(compare_name_df, 'García Stobart', 'Miguel Angel García Stobart' )
compare_name_df = change_player_values(compare_name_df, 'García Miguel Angel', 'Miguel Angel García Stobart' )

# Martínez
compare_name_df = change_player_values(compare_name_df, 'Martínez Daniel', 'Daniel Ginés Martínez' )

# Hernangómez
compare_name_df = change_player_values(compare_name_df, 'Hernangómez G.', 'Willy Hernangómez' )

# Sánchez
compare_name_df = change_player_values(compare_name_df, 'Sánchez Javier', 'Xavi Sánchez Bernat' )
compare_name_df = change_player_values(compare_name_df, 'Sánchez', 'Sergio Sánchez Pérez' )

# Sánchez
compare_name_df = change_player_values(compare_name_df, 'Sánchez Javier', 'Xavi Sánchez Bernat' )
compare_name_df = change_player_values(compare_name_df, 'Sánchez', 'Sergio Sánchez Pérez' )

# Sánchez
compare_name_df = change_player_values(compare_name_df, 'Van de Hare R.', 'Remon Van de Hare' )
compare_name_df = change_player_values(compare_name_df, 'Van Lacke Federico', 'Fede Van Lacke' )
compare_name_df = change_player_values(compare_name_df, 'Van den Spiegel T.', 'Tomas Van den Spiegel' )

# Toledo
compare_name_df = change_player_values(compare_name_df, 'Toledo M.', 'Marcus Toledo' )
compare_name_df = change_player_values(compare_name_df, 'Toledo Santiago', 'Santi Toledo' )

# Todorovic
compare_name_df = change_player_values(compare_name_df, 'Todorovic Marco', 'Marko Todorovic' )

# Davis
compare_name_df = change_player_values(compare_name_df, 'Davis Paul Russell', 'Paul Davis' )

# Ruiz
compare_name_df = change_player_values(compare_name_df, 'Ruiz de GalarretaA.', 'Alberto Ruiz de Galarreta' )

# Ruiz
compare_name_df = change_player_values(compare_name_df, 'Ruiz de GalarretaA.', 'Alberto Ruiz de Galarreta' )

# Rest
compare_name_df = change_player_values(compare_name_df, 'De la Fuente R.', 'Rodrigo de la Fuente')
compare_name_df = change_player_values(compare_name_df, 'Alvarez Berni', 'Berni Álvarez')
compare_name_df = change_player_values(compare_name_df, 'Mc Guthrie Chris', 'Chris McGuthrie')
compare_name_df = change_player_values(compare_name_df, 'Rodríguez Dani', 'Daniel Rodríguez')
compare_name_df = change_player_values(compare_name_df, 'Khansen T.', 'Travis Hansen')
compare_name_df = change_player_values(compare_name_df, 'Garcia Jojo', 'JoJo Roundy')
compare_name_df = change_player_values(compare_name_df, 'Williams J.', 'Jawad Williams')
compare_name_df = change_player_values(compare_name_df, 'Bivia Carles', 'Carles Bivià')
compare_name_df = change_player_values(compare_name_df, 'Stefansson Jon', 'Jón Stefánsson')
compare_name_df = change_player_values(compare_name_df, 'Dean Taquan', 'Taqwa Piñero')
compare_name_df = change_player_values(compare_name_df, 'Jelinek David', 'David Jelínek')
compare_name_df = change_player_values(compare_name_df, 'Lavrinovic', 'Darjus Lavrinovic')
compare_name_df = change_player_values(compare_name_df, 'Freire Luz Rafa', 'Rafa Luz')
compare_name_df = change_player_values(compare_name_df, 'Uriz M.', 'Mikel Úriz')
compare_name_df = change_player_values(compare_name_df, 'Williams', 'Latavious Williams')
compare_name_df = change_player_values(compare_name_df, 'Burstchi Jacob', 'Jacob Burtschi')
compare_name_df = change_player_values(compare_name_df, 'Diagné C.', 'Moussa Diagne')
compare_name_df = change_player_values(compare_name_df, 'OLeary Ian Joseph', 'Ian O´Leary')
compare_name_df = change_player_values(compare_name_df, 'Manny Quezada', 'Emmanuel Quezada')
compare_name_df = change_player_values(compare_name_df, 'Tony Gaffney', 'Gaffney Anthony')
compare_name_df = change_player_values(compare_name_df, 'Bitjaa Kody Johan', 'Johan Kody')
compare_name_df = change_player_values(compare_name_df, 'Rodríguez Manuel', 'Manu Rodríguez')

# Alter name of player
temp = result.copy()
player_map = map(lambda x: alter_player_name(x, compare_name_df, result) ,list(compare_name_df["player_name"]))
for i in player_map:
    print(i)

result.to_csv("db_with_fixed_player.csv", index=False)
df = pd.read_csv("db_with_fixed_player.csv")
for i in ["action","team","team_name","player_code","player_name"]:
    df.loc[:,i].fillna("NA", inplace = True)
    df[i] = df[i].astype('string')

# Alter players' name with the same name
# Sergio Rodríguez Gómez
mask = (player["player"] == "Sergio Rodríguez")&(player["age"] == 35)
new_name = "Sergio Rodríguez Gómez"
player.loc[mask, 'player'] = new_name
mask = df["player_name"] == "Sergio Rodríguez"
df.loc[mask, 'player_name'] = new_name

# Sergio Pérez Soriano
mask = (player["player"] == "Sergio Pérez")&(player["age"] == 37)
new_name = "Sergio Pérez Soriano"
player.loc[mask, 'player'] = new_name
mask = (df["player_name"] == "Sergio Pérez")&(df["team_name"] == "Akasvayu Girona" )
df.loc[mask, 'player_name'] = new_name

# Jorge García García
mask = (player["player"] == "Jorge García")&(player["age"] == 33)
new_name = "Jorge García García"
player.loc[mask, 'player'] = new_name

# Daniel López
mask = (player["player"] == "Daniel López")&(player["age"] == 33)
new_name = "Sergio Pérez Soriano"
player.loc[mask, 'player'] = new_name
mask = (df["player_name"] == "Daniel López")&(df["team_name"].str.contains("Manresa") )
df.loc[mask, 'player_name'] = new_name

# Jorge García García
mask = player["player"] == "Marko Popovic"
new_name = "Croacia"
player.loc[mask, 'nationality'] = new_name

# Zan Tabak
mask = player["player"] == "Zan Tabak"
height = 213
age = 51
temp_ = 2
player.loc[mask, "height"] = height
player.loc[mask, "age"] = age
player.loc[mask, "temp."] = temp_

## Alter position value that are duplicated by string, e.g. "AleroAlero" => "Alero"
position_lst_alter = ['BaseBase', 'AleroAlero', 'EscoltaEscolta', 'Ala-pívotAla-pívot', 'PívotPívot']
for pos in position_lst_alter:
    pos_len = int(len(pos)/2)
    mask = player["position"] == pos
    new_pos = pos[0:pos_len]
    player.loc[mask, 'position'] = new_pos

############
## Player ##
############
player.rename(columns = {"age": "current_age"}, inplace = True)
player.drop_duplicates(subset = ['player','year','team'], inplace = True)
player_df = player.groupby(['player']).count().reset_index().loc[:,['player']]

player_df = player_df.merge(player.groupby(['player','height']).count().reset_index().loc[:,['player', 'height']], how='left', on='player')
player_df = player_df.merge(player.groupby(['player','nationality']).count().reset_index().loc[:,['player', 'nationality']], how='left', on='player')
player_df = player_df.merge(player.groupby(['player','current_age']).count().reset_index().loc[:,['player', 'current_age']], how='left', on='player')


#######################
## player_year_group ##
#######################

# Fill missing current age and temp acording to multiple entries of a palyer (if exist)
pyg_df = player.loc[:, ['player', 'year', 'team', 'position', 'temp.', 'numberp', 'current_age']].copy()
frequant_player_lst = pyg_df.groupby(['player']).count().reset_index()
frequant_player_lst = list(frequant_player_lst[frequant_player_lst["year"] > 1].loc[:,"player"])
null_info_player_lst = list(pyg_df[pyg_df["current_age"].isnull()].loc[:,"player"])
list_of_name = [*frequant_player_lst, *null_info_player_lst]
list_of_name = [ name for name in set(list_of_name) if (name in frequant_player_lst)&(name in null_info_player_lst)]

th = []
start_time = time.time()
for ind, name in enumerate(list_of_name):
    thread = thrd.Thread(name='th%s' % ind, target=fill_missing_year_temp(name,pyg_df, ))
    thread.start()
    th.append(thread)
for j in th:
    j.join()
print(time.time() - start_time)

# AGE
pyg_df['age'] = pyg_df['current_age'] - (2022 - pyg_df['year'])
pyg_df.drop(columns='current_age', inplace = True)


##########
## team ##
##########

# Fix team name
th = []
start_time = time.time()
for new_team_name, team_list in const.constant.team_team_dict.items():
    thread = thrd.Thread( target=alter_team_name(new_team_name,team_list,df, ))
    thread.start()
    th.append(thread)
for j in th:
    j.join()
print(time.time() - start_time)

team_df = df.groupby(['team', 'team_name']).count().reset_index().loc[:,['team', 'team_name']].sort_values(by='team')


for team, stad in const.constant.team_stadium.items():
    mask = team_df["team_name"] == team
    team_df.loc[mask, "stadium"] = stad[0]

##########
## GAME ##
##########

game_index_lst = list(df[df['play_number'] == 2].index)
away_team_index_lst = [i+5 for i in game_index_lst]
game_df = pd.DataFrame()
game_df["game_id"] = df.iloc[game_index_lst, 1].reset_index(drop=True)
game_df["year"] = df.iloc[away_team_index_lst, 0].reset_index(drop=True)
game_df["home_team"] = df.iloc[game_index_lst, 5].reset_index(drop=True)
game_df["away_team"] = df.iloc[away_team_index_lst, 5].reset_index(drop=True)

temp = df.groupby(["year","game_id"]).max()["play_number"]
index_ = temp.index
end_game_index_lst = game_index_lst.copy()
end_game_index_lst.pop(0)
end_game_index_lst.append(df.shape[0]-2)
end_game_index_lst = [i-3 for i in end_game_index_lst]
max_hour = df.loc[end_game_index_lst, "timeStamp"]
min_hour = df.loc[game_index_lst, "timeStamp"]
game_df["start_game_time"] = pd.to_datetime(min_hour.values)
game_df["end_game_time"] = pd.to_datetime(max_hour.values)
game_df["game_length"] = round((game_df["end_game_time"]-game_df["start_game_time"]).dt.total_seconds()/3600,3)
game_df["game_length"] = game_df["game_length"].map('{:.2f}'.format)
game_df["game_length"] = game_df["game_length"].astype(float)

##After running this part please run "game length column again
# Fix datetime error
temp = game_df[game_df["game_length"]<0]

# 'year'] == 2004 "game_id"] == 184
mask = (game_df['year'] == 2004)&(game_df["game_id"] == 184)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2005-02-14 21:07:13')

# 'year'] == 2006 "game_id"] == 220
mask = (game_df['year'] == 2006)&(game_df["game_id"] == 220)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2007-03-19 14:53:40')

# 'year'] == 2006 "game_id"] == 243
mask = (game_df['year'] == 2006)&(game_df["game_id"] == 243)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2007-03-31 21:21:21')

# 'year'] == 2007 "game_id"] == 257
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 257)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-07-04 14:34:50')

# 'year'] == 2011 "game_id"] == 312
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 312)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-05-23 14:10:50')

# 'year'] == 2010 "game_id"] == 302
mask = (game_df['year'] == 2011)&(game_df["game_id"] == 302)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2012-06-05 19:43:27')

temp = game_df[game_df["game_length"]>5]

# 'year'] == 2003 "game_id"] == 67
mask = (game_df['year'] == 2003)&(game_df["game_id"] == 67)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2003-10-30 10:20:19')

# 'year'] == 2004 "game_id"] == 194
mask = (game_df['year'] == 2004)&(game_df["game_id"] == 194)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2005-02-28 14:19:32')

# 'year'] == 2006 "game_id"] == 15
mask = (game_df['year'] == 2006)&(game_df["game_id"] == 15)
game_df.loc[mask, 'start_game_time'] = pd.to_datetime('2006-07-10 14:06:07')

# 'year'] == 2006 "game_id"] == 21
mask = (game_df['year'] == 2006)&(game_df["game_id"] == 21)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2006-11-12 22:52:07')

# 'year'] == 2007 "game_id"] == 147
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 147)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-12-01 22:47:00')

# 'year'] == 2007 "game_id"] == 206
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 206)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-07-03 20:45:00')

# 'year'] == 2007 "game_id"] == 67
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 206)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-07-03 20:45:00')

# 'year'] == 2007 "game_id"] == 206
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 206)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-07-03 20:45:00')

# 'year'] == 2004 "game_id"] == 6
mask = (game_df['year'] == 2004)&(game_df["game_id"] == 6)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2004-03-10 20:15:00')
game_df.loc[mask, 'start_game_time'] = pd.to_datetime('2004-03-10 18:16:25')

# 'year'] == 2004 "game_id"] == 194
mask = (game_df['year'] == 2004)&(game_df["game_id"] == 194)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2005-02-27 14:19:32')

# 'year'] == 2006 "game_id"] == 21
mask = (game_df['year'] == 2006)&(game_df["game_id"] == 21)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2006-11-10 22:52:00')

# 'year'] == 2007 "game_id"] == 147
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 147)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-01-13 22:03:00')
# 'year'] == 2007 "game_id"] == 206
mask = (game_df['year'] == 2007)&(game_df["game_id"] == 206)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2008-02-03 20:45:00')
# 'year'] == 2010 "game_id"] == 6
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 6)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2010-02-10 21:49:03')
# 'year'] == 2010 "game_id"] == 85
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 85)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2010-05-12 14:34:40')
# 'year'] == 2010 "game_id"] == 128
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 128)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-08-01 21:57:02')
# 'year'] == 2010 "game_id"] == 197
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 187)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-02-20 21:30:19')
# 'year'] == 2010 "game_id"] == 206
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 206)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-06-03 14:15:12')
# 'year'] == 2010 "game_id"] == 231
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 231)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-03-20 16:44:44')
# 'year'] == 2010 "game_id"] == 252
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 252)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-03-04 14:12:32')
# 'year'] == 2010 "game_id"] == 258
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 258)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-10-04 14:05:43')
# 'year'] == 2010 "game_id"] == 286
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 286)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-01-05 21:36:08')
# 'year'] == 2010 "game_id"] == 291
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 291)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-04-05 21:44:12')
# 'year'] == 2010 "game_id"] == 95
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 95)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2010-12-12 14:07:12')
# 'year'] == 2010 "game_id"] == 95
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 127)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-09-01 14:20:33')
# 'year'] == 2010 "game_id"] == 95
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 158)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-01-23 14:06:09')
# 'year'] == 2010 "game_id"] == 169
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 169)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-01-29 19:46:20')
game_df.loc[mask, 'start_game_time'] = pd.to_datetime('2011-01-29 17:49:29')
# 'year'] == 2010 "game_id"] == 174
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 174)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-06-02 14:40:22')
# 'year'] == 2010 "game_id"] == 185
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 185)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-02-02 14:14:51')
# 'year'] == 2010 "game_id"] == 189
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 189)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-02-20 13:54:14')
# 'year'] == 2010 "game_id"] == 204
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 204)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-06-03 14:01:55')
# 'year'] == 2010 "game_id"] == 207
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 207)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-06-03 13:44:48')
# 'year'] == 2010 "game_id"] == 223
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 223)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-03-13 14:19:15')
# 'year'] == 2010 "game_id"] == 231
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 231)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-03-20 14:41:19')
# 'year'] == 2010 "game_id"] == 262
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 262)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-04-17 14:11:07')
# 'year'] == 2010 "game_id"] == 281
mask = (game_df['year'] == 2010)&(game_df["game_id"] == 281)
game_df.loc[mask, 'end_game_time'] = pd.to_datetime('2011-01-05 14:18:34')

# Run it after run evrything else
# 'year'] == 2004 "game_id"] == 15
mask = (game_df['year'] == 2004)&(game_df["game_id"] == 15)
game_df.loc[mask, 'game_length'] = round(74/60,2)

# score
temp = df.loc[:, ['year', 'game_id', 'team', 'action']].assign(result1 = np.where(df['action']=="Canasta de 1",1,0),
                                                                    result2 = np.where(df['action']=="Canasta de 2",2,0),
                                                                    result3 = np.where(df['action']=="Canasta de 3",3,0)
                                                                    ).groupby(['year', 'game_id', 'team']).agg({'result1':sum, 'result2':sum, 'result3':sum})
temp.reset_index(inplace=True)
temp = temp[temp["team"] != 'NA']
temp['score'] = temp['result1'] + temp['result2'] + temp['result3']
temp_1 = game_df.copy()

game_df = game_df.merge(temp.loc[:,["game_id","year","team", "score"]], left_on = ["game_id","year","home_team"], right_on=["game_id","year","team"]).drop(columns = ['team'])
game_df.rename(columns={"score":"home_score"}, inplace=True)
game_df = game_df.merge(temp.loc[:,["game_id","year","team", "score"]], left_on = ["game_id","year","away_team"], right_on=["game_id","year","team"]).drop(columns = ['team'])
game_df.rename(columns={"score":"away_score"}, inplace=True)

game_df["home_win"] = game_df["home_score"] > game_df["away_score"]

#####################################################################################

# SQL EXAMINATION, STOP PROCESSING THE DATA, WE HAVE THE 5 BASIC TABLES
# JUST CHECK DATA TYPES

#####################################################################################

# GAME INFO - CHECK
game_df.to_csv("game_df.csv", index=False)

# PLAYER_DF - CHECK
player_df['player'] = player_df['player'].astype('string')
player_df['nationality'] = player_df['nationality'].astype('string')
player_df.to_csv("player_df.csv", index=False)

# PYG  - CHECK
pyg_df['player'] = pyg_df['player'].astype('string')
pyg_df['team'] = pyg_df['team'].astype('string')
pyg_df['position'] = pyg_df['position'].astype('string')
pyg_df.to_csv("pyg_df.csv", index=False)

# TEAM_DF  - CHECK
team_df['stadium'] = team_df['stadium'].astype('string')
team_df.to_csv("team_df.csv", index=False)

#############################
## PLAYER GAME PERFORMANCE ##
#############################







