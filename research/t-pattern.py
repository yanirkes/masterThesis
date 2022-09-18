import random
import numpy as np
import pandas as pd
from more_itertools import locate
import itertools as itr
from scipy.stats import binom

def calc_diff(i, data):
    print(i)
    if i == 0:
        return data.iloc[i,0] - 0
    else:
        return data.iloc[i, 0] - data.iloc[i-1, 0]

def calc_time_diff_two_events(i_1, i_2, data):
    return data.iloc[i_2, 0] - data.iloc[i_1 - 1, 0]

def calc_statistic(data, N, sequence, Nab, first_event, second_event):
    print(data,'\n####', N,'\n####', sequence,'\n####', Nab,'\n####', first_event,'\n####', second_event)
    d = abs(data[1]-data[0])
    print(d)
    p_hat = 1 - sequence.count(second_event)/N
    print(p_hat)
    Na = sequence.count(first_event)
    print(Na)
    p_val = 1-sum([ binom.pmf(i, Na, np.power(p_hat,d)) for i in range(0,Nab-1)])
    print( np.power(p_hat,d))
    return {"seq" : first_event+'_'+second_event,
        "d_i" : d,
        "p_val" : p_val}



n = 7
time = np.round(np.sort(np.random.normal(60,30,(1,n))))

sequence = ["a","b","f","a","b","a","b"]

df = pd.DataFrame(columns=["time", "sequence"])

for x in time.transpose().tolist():
    print(x)

df["time"] = [x[0] for x in time.transpose().tolist()]

df["sequence"] = [x for x in sequence]

df['diff'] = [calc_diff(x, df) for x in range(0,n)]


df_time_dist = pd.DataFrame(columns=["first", "second", "sequence"])
# Over all the permutations calc the distance if FIRST event happened before the SECOND event
for i in itr.permutations(set(sequence),2):
    print(i)
    # Take all the indices of event first and second
    first_e = list(locate(sequence, lambda x: x == i[0]))
    second_e = list(locate(sequence, lambda x: x == i[1]))
    # print(first_e,'\n',second_e)
    # For all FIRST event check if index is sequential and append line
    for a in first_e:
        for b in second_e:
            if b>a:
                # print(a, b)
                series_ = pd.Series([df.iloc[a,0],df.iloc[b,0],i[0] +"_"+ i[1] ], index=["first","second", "sequence"])
                print(series_)
                df_time_dist = df_time_dist.append(series_, ignore_index = True)
                break
df_time_dist['delta_time'] = df_time_dist['second'] - df_time_dist['first']

statistic_dist_calc = pd.DataFrame(columns=["sequence","d_i", "p_hat"])
for i in set(df_time_dist['sequence']):
    temp = df_time_dist[df_time_dist['sequence'] == i]
    # break
    if temp.shape[0] <= 1:
        val = {"seq" : i,
        "d_i" : 100000,
        "p_hat" : 1}
        # statistic_dist_calc = pd.concat([statistic_dist_calc,val])
    else:
        events = i.split('_')
        event_a = events[0]
        event_b = events[1]
        Nab = temp.shape[0]
        val = calc_statistic(temp.loc[:,'delta_time'], 100, sequence,  Nab, event_a, event_b )
        # statistic_dist_calc = pd.concat([statistic_dist_calc, val])
        break

        # def calc_statistic(data, N, sequence, Nab, first_event, second_event):