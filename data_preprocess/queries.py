import utils.postgres_config as psql
import utils.constant_class as con
import pandas as pd


dbConn_obj = psql.PostgressCon()
###################################
## DATABASE CONSTRUCTION QUERIES ##
###################################

# GAME INFO - CHECK
game_df = pd.read_csv("game_df.csv")
dbConn_obj.table_from_df(game_df,"basket", "game_info")

# PLAYER_DF - CHECK
player_df = pd.read_csv("player_df.csv")
dbConn_obj.table_from_df(player_df,"basket", "player_info")

# PYG  - CHECK
pyg_df = pd.read_csv("pyg_df.csv")
dbConn_obj.table_from_df(pyg_df,"basket", "player_year_group")

# TEAM_DF  - CHECK
team_df = pd.read_csv("team_df.csv")
dbConn_obj.table_from_df(team_df,"basket", "team_info")

# STG - Full data base with fixed value
stg_df = pd.read_csv("csv_files/db_with_fixed_player.csv")
for i in ["action","team","team_name","player_code","player_name"]:
    stg_df.loc[:,i].fillna("NA", inplace = True)
    stg_df[i] = stg_df[i].astype('string')

stg_df["timeStamp"] = pd.to_datetime(stg_df["timeStamp"])
stg_df.rename(columns={"timeStamp": "tmstmp"}, inplace = True)
dbConn_obj.table_from_df(stg_df,"stg", "all_data_info_2")
