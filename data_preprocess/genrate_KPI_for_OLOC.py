import pandas as pd
import multiprocessing as mp
from collections import Counter
from itertools import chain
from itertools import product
import utils.postgres_config as psql
import math
import numpy as np

q = """
select distinct year, game_id, time_marker, score_dif
from basket.data_analytics_all_metrics_agg
order by 1,2,3 asc
"""
con = psql.PostgressCon()
con.connect()
df = con.execute_query_with_headers(q)
df = pd.DataFrame(data = df[0], columns=df[1])

abs_temp = df.copy()
abs_temp.score_dif = abs(abs_temp.score_dif )
abs_temp_g = abs_temp.groupby(['year','game_id']).agg({'score_dif':['mean',np.std]}).reset_index()
abs_temp_g['cv'] = abs_temp_g[('score_dif',  'std')]/abs_temp_g[('score_dif', 'mean')]
abs_temp_g = abs_temp_g.sort_values(by='cv')
abs_temp_g.columns = abs_temp_g.columns.droplevel(0)
abs_temp_g.columns = ['year','game_id','mean_score_dif','std_score_dif','cv_score_dif']
abs_temp_g.replace([np.inf, -np.inf], np.nan,inplace=True)
abs_temp_g.fillna(500, inplace=True)

abs_temp_g.hist(bins=20)


con.table_from_df(abs_temp_g,"basket", "abs_score_dif_cv_per_game")

temp = df.groupby(['year','game_id']).agg({'score_dif':['mean',np.std]}).reset_index()
temp['cv'] = temp[('score_dif',  'std')]/temp[('score_dif', 'mean')]
temp = temp.sort_values(by='cv')
temp.columns = temp.columns.droplevel(0)
temp.columns = ['year','game_id','mean_score_dif','std_score_dif','cv_score_dif']
temp.replace([np.inf, -np.inf], np.nan,inplace=True)
temp.fillna(500, inplace=True)

temp.loc[abs(temp.cv_score_dif) < 50,:].cv_score_dif.hist(bins=100)


con.table_from_df(temp,"basket", "score_dif_cv_per_game")