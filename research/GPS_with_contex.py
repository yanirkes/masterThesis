import utils.postgres_config as psql
import utils.constant_class as con
import pandas as pd
import time

#################
# PREPROCESSING #
#################

dbConn_obj = psql.PostgressCon()

def lineup(df, sub, x):
    print("year: %s , game_id: %s, team: %s" % (x[0], x[1], x[2]))
    lst = []
    ndf = df[(df['year'] == x[0])&(df['game_id'] == x[1] )&(df['team'] == x[2] )].reset_index(drop=True)
    new_lineup = ndf.loc[0, 'lineup']
    p_out = ndf.loc[0, 'player_out']
    p_in = ndf.loc[0, 'player_in']
    ind = 0

    # Check if in this chunck, there are more than 5 substitution at the begin of a quarter
    if not sub[(sub['year'] == x[0]) & (sub['game_id'] == x[1]) & (sub['team'] == x[2])].empty:
        time_marker = sub.loc[(sub['year'] == x[0]) & (sub['game_id'] == x[1]) & (sub['team'] == x[2]), 'time_marker_origin'].values[0]
        occurance = sub.loc[(sub['year'] == x[0]) & (sub['game_id'] == x[1]) & (sub['team'] == x[2]), 'c'].values[0]
    else:
        time_marker = 100000
        occurance = 100000
    return  recursive_lineup(ndf ,new_lineup ,p_out ,p_in ,ind, lst, time_marker, occurance)


def recursive_lineup_special(df, new_lineup, p_out, p_in, ind, lst):
    new_lineup_ = new_lineup.replace(p_out, p_in )
    lst.append(new_lineup_)
    if ind >= df.shape[0]-1:
        if (len(lst) != df.shape[0]):
            print('###########################################################################', '\n', lst, '\n', df.to_string())
            # raise ValueError("Len not equal", len(lst), ' ',df.shape[0])
        return lst
    ind +=1
    p_out = df.loc[ind, 'player_out']
    p_in = df.loc[ind, 'player_in']
    return recursive_lineup_special(df, new_lineup_, p_out, p_in, ind,lst)

def recursive_lineup(df, new_lineup, p_out, p_in, ind, lst, time_marker=100000000, occurance = 5):
    time = df.loc[ind, 'time_marker_origin']
    # print(time,' ',df[df['time_marker_origin'] == time].shape[0])
    if (df.loc[ind, 'time_marker_origin'] in [0, 600, 1200, 1800, 2400])&(df[df['time_marker_origin'] == time].shape[0] > 1):
        if df.loc[ind, 'time_marker_origin'] == time_marker:
            # First 5 players
            temp_df = df.sort_values(by='play_number_entre_sale').reset_index(drop=True)
            ind, new_lineup_ = replace_quarterly_lineup(temp_df, ind)
            lst = lst + new_lineup_

            # Last players in the substitition list
            ind += 1
            size_of_new_df = occurance - 5
            temp_df = temp_df.loc[ind:ind + size_of_new_df - 1, :].reset_index(drop=True)
            p_out = temp_df.loc[0, 'player_out']
            p_in = temp_df.loc[0, 'player_in']
            lst_tmp = []
            lst_tmp = recursive_lineup_special(temp_df, new_lineup_[0], p_out, p_in, 0, lst_tmp)
            ind += size_of_new_df-1
            lst = lst + lst_tmp
            new_lineup_ = lst_tmp[-1]
        else:
            ind, new_lineup_ = replace_quarterly_lineup(df, ind)
            lst = lst + new_lineup_
            new_lineup_ = new_lineup_[0]
    else:
        new_lineup_ = new_lineup.replace(p_out, p_in)
        lst.append(new_lineup_)
    if ind >= df.shape[0] - 1:
        if len(lst) != df.shape[0]:
            print('###########################################################################', '\n', lst, '\n', df.to_string())
            # raise ValueError("Len not equal", len(lst), ' ',df.shape[0])
        return lst
    ind += 1
    p_out = df.loc[ind, 'player_out']
    p_in = df.loc[ind, 'player_in']
    return recursive_lineup(df, new_lineup_, p_out, p_in, ind, lst, time_marker, occurance)

def replace_quarterly_lineup(df, ind):
    replication_number = df.shape[0] - ind if ind + 4 > df.shape[0] else 4
    new_lineup = df.loc[ind:ind + replication_number, 'player_in'].tolist()
    new_lineup_str = ','.join(new_lineup)
    lst = [new_lineup_str for i in range(0, replication_number + 1)]
    ind = ind + replication_number
    return ind, lst

# CREATE LINEUP AND SUB LINEUP
st = time.time()
q = """
select a_.year, a_.game_id, a_.team, a_.time_marker, a_.time_marker_origin, a_.play_number_entre_sale, player_out,player_in, lineup  as lineup
from
(
select *
from basket.substitition
where  player_out != 'openning 5'
	and player_in != 'closing 5'
order by year, game_id, time_marker asc
 ) as a_
left join 
(select year, game_id, team, lineup
 from basket.stating_lineup
 ) as b_
on a_.game_id = b_.game_id
and a_.year = b_.year
and a_.team = b_.team
order by 1,2,3,5 asc
"""

q2 = """
with tem as(
select a_.year, a_.game_id, a_.team, a_.time_marker, a_.time_marker_origin, player_out,player_in, lineup
from
(
select *
from basket.substitition
where  player_out != 'openning 5'
	and player_in != 'closing 5'
order by year, game_id, time_marker asc
 ) as a_
left join 
(select year, game_id, team, lineup
 from basket.stating_lineup
 ) as b_
on a_.game_id = b_.game_id
and a_.year = b_.year
and a_.team = b_.team
order by 1,2,3,5 asc
	)
 select year, game_id, time_marker, time_marker_origin,team, count(1) as c
from tem
where (time_marker in (0,600,1200,1800,2400) or (game_id = 321 and year = 2006 and team ='JOV'))
group by 1,2,3,4,5
having count(1)  > 5
order by year, game_id, time_marker"""

lineup_df = dbConn_obj.execute_query_with_headers(q)
lineup_df = pd.DataFrame(data = lineup_df[0], columns = lineup_df[1])
over_frequency_info_df =  dbConn_obj.execute_query_with_headers(q2)
over_frequency_info_df = pd.DataFrame(data = over_frequency_info_df[0], columns = over_frequency_info_df[1])

opening_lineup = lineup_df.groupby(['year','game_id','team']).head(1).reset_index(drop=True)
opening_lineup['time_marker'] = 0
opening_lineup['time_marker_origin'] = 0
opening_lineup['player_out'] = '-'
opening_lineup['player_in'] = '-'

lst_game_year_team = lineup_df[['year','game_id', 'team']].reset_index(drop=True)
lst_game_year_team = lst_game_year_team.groupby(['year','game_id','team']).max().reset_index()
# lst_game_year_team = lst_game_year_team.iloc[0:12,:]
# lst_game_year_team = lst_game_year_team[(lst_game_year_team['year'] == 2003)&(lst_game_year_team['game_id'].isin([91]))&(lst_game_year_team['team'].isin(['ALI']))]
# lineup_df_t = lineup_df.iloc[:12,:]
# lineup_df_t = lineup_df[(lineup_df['year'] == 2003)&(lineup_df['game_id'].isin([91]))&(lineup_df['team'].isin(['ALI']))]

lst = []
for x in lst_game_year_team.values:
    t = lineup(lineup_df, over_frequency_info_df, x)
    print(len(t))
    lst = lst + t

# temp = map(lambda x: lineup(lineup_df,over_frequency_info_df,x ), lst_game_year_team.values)
#
# lst = []
# for ind, i in enumerate(temp):
#     key = lst_game_year_team.iloc[ind,:]
#     len_ = lineup_df[(lineup_df['year'] == key[0])&(lineup_df['game_id'] == key[1] )&(lineup_df['team'] == key[2] )].shape[0]
#     if len_ != len(i):
#         print("########\n###########\n############\n########\nThe len is", len_,' vs ',len(i))
#     lst = lst + i

opening_lineup['new_lineup'] = opening_lineup['lineup']
lineup_df['new_lineup'] = lst
lineup_df = pd.concat([lineup_df, opening_lineup], axis=0).reset_index(drop=True)
lineup_df = lineup_df.sort_values(by=['year','game_id','team','time_marker'])
lineup_df = lineup_df.reset_index(drop=True)
print(time.time() - st)

dbConn_obj.table_from_df(lineup_df,"basket", "lineup_with_subtitution_temp")


for i in lst[83639:83652]:
    print(i)

for i in lst[43970:43979]:
    print(i)