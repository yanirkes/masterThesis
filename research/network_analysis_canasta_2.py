import pandas as pd
import networkx as nt
import numpy as np
import utils.postgres_config as psql

data = pd.read_csv("csv_files/canasta_2_player_relation.csv")
player = pd.read_csv("csv_files/player_info_2003.csv")
gsp_data = pd.read_csv("csv_files/GSP_2003_pattern_with_players.csv")

# avoid na
data = data.fillna('NA')
gsp_data = gsp_data.fillna('NA')

# Create mapping
edges_tMinus2_tMinus1 = pd.DataFrame(columns=player.player, index=player.player)
edges_tMinus2_tMinus1 = edges_tMinus2_tMinus1.fillna(0)
# edges_tMinus1_t = pd.DataFrame(columns=player.player, index=player.player)
# edges_tMinus1_t = edges_tMinus1_t.fillna(0)

# creating relation
for player_minus2, player_minus1 in zip(data.player_name_minus_2,data.player_name_minus_1):
    if (player_minus2 != 'NA')&(player_minus1 != 'NA'):
        edges_tMinus2_tMinus1.loc[player_minus2, player_minus1] = edges_tMinus2_tMinus1.loc[player_minus2, player_minus1]+1

for player_minus1, player_minus in zip(data.player_name_minus_1,data.player_name):
    if (player_minus1 != 'NA')&(player_minus != 'NA'):
        edges_tMinus2_tMinus1.loc[player_minus1, player_minus] = edges_tMinus2_tMinus1.loc[player_minus1, player_minus]+1

edges_tMinus2_tMinus1.to_csv('csv_files/network_canasta_2_edges.csv')

# From gephi, add more info
player_after_gephi = pd.read_csv("csv_files/canasta_2_player_relation_from_gephi.csv")
player_after_gephi = player_after_gephi.drop(['team','position'], axis=1)
player_after_gephi  = player_after_gephi.merge(player, left_on = 'Id', right_on='player')
player_after_gephi = player_after_gephi.drop(['numberp','year'], axis=1)
player_after_gephi.to_csv('csv_files/network_canasta_2_nodes_indo.csv')


# creating relation with GSP algorithm
for player_minus3, player_minus2 in zip(gsp_data.player_name_minus_2,gsp_data.player_name_minus_1):
    if (player_minus3 != 'NA')&(player_minus2 != 'NA'):
        edges_tMinus2_tMinus1.loc[player_minus3, player_minus2] = edges_tMinus2_tMinus1.loc[player_minus3, player_minus2]+1

for player_minus2, player_minus1 in zip(gsp_data.player_name_minus_2,gsp_data.player_name_minus_1):
    if (player_minus2 != 'NA')&(player_minus1 != 'NA'):
        edges_tMinus2_tMinus1.loc[player_minus2, player_minus1] = edges_tMinus2_tMinus1.loc[player_minus2, player_minus1]+1

for player_minus1, player_minus in zip(gsp_data.player_name_minus_1,gsp_data.player_name):
    if (player_minus1 != 'NA')&(player_minus != 'NA'):
        edges_tMinus2_tMinus1.loc[player_minus1, player_minus] = edges_tMinus2_tMinus1.loc[player_minus1, player_minus]+1
# filter low weight
edges_tMinus2_tMinus1 = edges_tMinus2_tMinus1.replace({1:0})

# filter self transition
for playar_ in player.player:
    edges_tMinus2_tMinus1.loc[playar_, playar_] = 0


edges_tMinus2_tMinus1.to_csv('csv_files/network_canasta_2_edges_gsp.csv')

# Network canasta number 2

q = """with temp_ as 
	(select  action
	, t_minus_1
	, t_minus_2
	, t_minus_3
	, player_name_team
	, player_name_minus_1
	, player_name_minus_2
	, player_name_minus_3
	, player_name_minus_4
	, player_name_minus_5
	, player_team_minus_1
	, player_team_minus_2
	, player_team_minus_3
	, player_team_minus_4
	, player_team_minus_5
	,count(1) frequency
	from
	(
		select game_id, action, 
		  case
  			when team = 'MAD' then player_name 
  			else team 
		  end as player_name_team
		, lag(action, 1) over(partition by year, game_id order by play_number asc) t_minus_1
		, lag(player_name, 1) over(partition by year, game_id order by play_number asc) player_name_minus_1
		, case 
			when lag(team, 1) over(partition by year, game_id order by play_number asc) = 'MAD' then lag(player_name, 1) over(partition by year, game_id order by play_number asc)
			else lag(team, 1) over(partition by year, game_id order by play_number asc)
		  end as player_team_minus_1
		, lag(action, 2) over(partition by year, game_id order by play_number asc) t_minus_2
		, lag(player_name, 2) over(partition by year, game_id order by play_number asc) player_name_minus_2
		, case 
			when lag(team, 2) over(partition by year, game_id order by play_number asc) = 'MAD' then lag(player_name, 2) over(partition by year, game_id order by play_number asc)
			else lag(team, 2) over(partition by year, game_id order by play_number asc)
		  end as player_team_minus_2
		, lag(action, 3) over(partition by year, game_id order by play_number asc) t_minus_3
		, lag(player_name, 3) over(partition by year, game_id order by play_number asc) player_name_minus_3
		, case 
			when lag(team, 3) over(partition by year, game_id order by play_number asc) = 'MAD' then lag(player_name, 3) over(partition by year, game_id order by play_number asc)
			else lag(team, 3) over(partition by year, game_id order by play_number asc)
		  end as player_team_minus_3
		, lag(action, 4) over(partition by year, game_id order by play_number asc) t_minus_4
		, lag(player_name, 4) over(partition by year, game_id order by play_number asc) player_name_minus_4
		, case 
			when lag(team, 4) over(partition by year, game_id order by play_number asc) = 'MAD' then lag(player_name, 4) over(partition by year, game_id order by play_number asc)
			else lag(team, 4) over(partition by year, game_id order by play_number asc)
		  end as player_team_minus_4
		, lag(action, 5) over(partition by year, game_id order by play_number asc) t_minus_5
		, lag(player_name, 5) over(partition by year, game_id order by play_number asc) player_name_minus_5, case 
			when lag(team, 5) over(partition by year, game_id order by play_number asc) = 'MAD' then lag(player_name, 5) over(partition by year, game_id order by play_number asc)
			else lag(team, 5) over(partition by year, game_id order by play_number asc)
		  end as player_team_minus_5
		, lag(action, 6) over(partition by year, game_id order by play_number asc) t_minus_6
		from stg.all_data_info
		where year = 2003
				AND time_marker not in (0, 600,1200,1800, 2400)
				and action not in  ('MARCA DE MINUTO','Sale a Banquillo','Entra a Pista', 'FIN DE PERIODO','Tiempo de TV')
				and game_id in (select distinct game_id from basket.game_info where year =2003 and (home_team = 'MAD' or away_team = 'MAD'))
	) as a_
	where
	  (action = 'Canasta de 2' and t_minus_1 = 'Falta recibida' and t_minus_2 =  'Falta Personal' and t_minus_3 = 'Entra a Pista')
	 or (action = 'Canasta de 2' and t_minus_1 in ('Intento fallado de 3','Intento fallado de 2') and t_minus_2 in ('Rebote Defensivo','Rebote Oefensivo') and t_minus_3 in ('Intento fallado de 3','Intento fallado de 2'))
	 or (action = 'Canasta de 2' and t_minus_1  in ('Intento fallado de 3','Intento fallado de 2') and t_minus_2 in ('Rebote Defensivo','Rebote Oefensivo') and t_minus_3 ='Pérdida')
	 or (action = 'Canasta de 2' and t_minus_1 in ('Rebote Defensivo','Rebote Oefensivo') and t_minus_2 in ('Intento fallado de 3','Intento fallado de 2') and t_minus_3 in ('Rebote Defensivo','Rebote Oefensivo')) 
	 group by  action
	, t_minus_1
	, t_minus_2
	, t_minus_3
	, player_name_team
	, player_name_minus_1
	, player_name_minus_2
	, player_name_minus_3
	, player_name_minus_4
	, player_name_minus_5
	, player_team_minus_1
	, player_team_minus_2
	, player_team_minus_3
	, player_team_minus_4
	, player_team_minus_5
	order by 14 desc
	) 	
	select action
	, t_minus_1
	, t_minus_2
	, t_minus_3
	, player_name_team
	, player_name_minus_1
	, player_name_minus_2
	, player_name_minus_3
	, player_team_minus_1
	, player_team_minus_2
	, player_team_minus_3
	, frequency
	, (select sum(frequency) from temp_) as N
	, round(frequency/(select sum(frequency) from temp_),4) as ratio_per_pattern /*sum to 1*/
	, (select count(distinct game_id) from stg.all_data_info where year = 2003)
	, frequency/cast((select count(distinct game_id) from stg.all_data_info where year = 2003) as float) as ratio_per_game /*sum all > 1*/
from temp_
order by 8 desc"""

q2 = """select distinct team
from stg.all_data_info
where year = 2003"""

q3 = """select distinct player_name
from stg.all_data_info
where year = 2003 
and team = 'MAD'
and player_name != 'NA'"""

con = psql.PostgressCon()
con.connect()
df = con.execute_query_with_headers(q)
team = con.execute_query_with_headers(q2)
player = con.execute_query_with_headers(q3)

df = pd.DataFrame(data=df[0], columns=df[1])
team = pd.DataFrame(data=team[0], columns=team[1])
player = pd.DataFrame(data=player[0], columns=player[1])
team.columns = ["node"]
player.columns = ["node"]
nodes = pd.concat([player,team], axis=0)

edges = pd.DataFrame(0,columns=list(nodes.node), index=list(nodes.node))

for player_minus3, player_minus2 in zip(df.player_team_minus_3,df.player_team_minus_2):
        edges.loc[player_minus3, player_minus2] = edges.loc[player_minus3, player_minus2]+1

for player_minus2, player_minus1 in zip(df.player_team_minus_2,df.player_team_minus_1):
        edges.loc[player_minus2, player_minus1] = edges.loc[player_minus2, player_minus1]+1

for player_minus1, player_minus in zip(df.player_team_minus_1,df.player_name_team):
        edges.loc[player_minus1, player_minus] = edges.loc[player_minus1, player_minus]+1

edges.to_csv('csv_files/mad_vs_all.csv')

# filter low weight
edges_tMinus2_tMinus1 = edges_tMinus2_tMinus1.replace({1:0})
