
import pandas as pd
import multiprocessing as mp
from collections import Counter
from itertools import chain
from itertools import product
import utils.postgres_config as psql
import time
import math
from scipy.stats import poisson,nbinom
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

class GSP:

    def __init__(self, raw_transactions):
        self.freq_patterns = []
        self._pre_processing(raw_transactions)

    def _pre_processing(self, raw_transactions):
        '''
        Prepare the data
        Parameters:
                raw_transactions: the data that it will be analysed
        '''
        self.max_size = max([len(item) for item in raw_transactions])
        self.transactions = [tuple(list(i)) for i in raw_transactions]
        counts = Counter(chain.from_iterable(raw_transactions))
        self.unique_candidates = [tuple([k]) for k, c in counts.items()]

    def _is_slice_in_list(self, s, l):
        len_s = len(s)  # so we don't recompute length of s on every iteration
        # print(l)
        return any(s == l[i:len_s + i] for i in range(len(l) - len_s + 1))


    def _calc_frequency(self, results, item):
        frequency = len([t for t in self.transactions if self._is_slice_in_list(item, t)])
        results[item] = frequency
        return results

    # def _calc_frequency(self, results, item, minsup):
    #     print(item)
    #     frequency = len([t for t in self.transactions if self._is_slice_in_list(item, t)])
    #     if frequency >= minsup:
    #         print(frequency)
    #         return frequency
    #     return 0


        #
        # temp = pd.DataFrame(self.transactions)
        # temp['ind'] = [list(temp.loc[i, :]) for i in temp.index]
        # temp = temp['ind']
        #
        # print(temp.applymap(lambda x: self._is_slice_in_list(item,x)))
        # # print(sum(temp['frequency']))
        #
        # # The number of times the item appears in the transactions
        # frequency = len(
        #     [t for t in self.transactions if self._is_slice_in_list(item, t)])
        # # frequency = len(
        # #     [t for t in self.transactions if set(list(item)).issubset(t)])
        # if frequency > minsup:
        #     results[item] = frequency
        # return results

    def _support(self, items):
        '''
        The support count (or simply support) for a sequence is defined as
        the fraction of total data-sequences that "contain" this sequence.
        (Although the word "contains" is not strictly accurate once we
        incorporate taxonomies, it captures the spirt of when a data-sequence
        contributes to the support of a sequential pattern.)
        Parameters
                items: set of items that will be evaluated
                minsup: minimum support
        '''
        results = mp.Manager().dict()
        pool = mp.Pool(processes=mp.cpu_count())

        for item in items:
            pool.apply_async(self._calc_frequency,
                             args=(results, item))
        pool.close()
        pool.join()

        return dict(results)

    # def _support(self, items):
    #     '''
    #     The support count (or simply support) for a sequence is defined as
    #     the fraction of total data-sequences that "contain" this sequence.
    #     (Although the word "contains" is not strictly accurate once we
    #     incorporate taxonomies, it captures the spirt of when a data-sequence
    #     contributes to the support of a sequential pattern.)
    #     Parameters
    #             items: set of items that will be evaluated
    #             minsup: minimum support
    #     '''
    #     results = {}
    #
    #     for item in items:
    #         results[item] = self._calc_frequency( item)
    #     print("The len of the new frequency dictionary")
    #     print(len(results.keys()))
    #     return results

    def _print_status(self, run, candidates):
        print("""
        Run {}
        There are {} candidates.
        The candidates have been filtered down to {}.\n"""
                      .format(run,
                              len(candidates),
                              len(self.freq_patterns[run - 1])))

    def calc_minsup(self,q, minsup_threshold,k_items):
        lambda_hat = np.mean(list(self.item_suppocrt_dict.values()))
        # quntile = poisson._ppf(q, lambda_hat)

        quntile = nbinom._ppf(q = q, n = k_items ,p = k_items/lambda_hat)
        minsup =  quntile if quntile > minsup_threshold else minsup_threshold
        print("The lambda and  minsup are:")
        print(lambda_hat, minsup)
        return minsup

    def filter_candidates(self, q, minsup_threshold, k_items):
        minsup = self.calc_minsup(q, minsup_threshold,k_items)
        temp_dict =  {k:v for (k,v) in self.item_suppocrt_dict.items() if v >= minsup}
        print("DONE create the new FILTERED frequency dictionary")
        return temp_dict

    @staticmethod
    def iter_lst(lst, ite):
        return [p + (ite,) for p in lst]

    def filter_items_in_candidates(self,k_items, items):
        candidates_lst = list(map(lambda x: self.iter_lst(list(set(self.freq_patterns[k_items - 2].keys())), x), items))
        return list(chain.from_iterable(candidates_lst))


    def search(self, minsup_threshold=50, q_for_first_run = 0.85, q = 0.95):
        '''
        Run GSP mining algorithm
        Parameters
                minsup: minimum support
        '''
        # assert (0.0 < minsup) and (minsup <= 1.0)
        # minsup = len(self.transactions) * minsup
        print('len trans: ', len(self.transactions))

        # the set of frequent 1-sequence: all singleton sequences
        # (k-itemsets/k-sequence = 1) - Initially, every item in DB is a
        # candidate
        candidates = self.unique_candidates


        # scan transactions to collect support count for each candidate
        # sequence & filter
        # originial
        # self.freq_patterns.append(self._support(candidates, minsup))
        self.item_suppocrt_dict = self._support(candidates)
        self.freq_patterns.append(self.filter_candidates(q_for_first_run, minsup_threshold, 1))


        # (k-itemsets/k-sequence = 1)
        k_items = 1

        self._print_status(k_items, candidates)

        # repeat until no frequent sequence or no candidate can be found
        while (k_items + 1 <= 5):
            print("Run number %s", k_items)
            # minsup = len(self.transactions)/(math.factorial(k_items)) if k_items > 3 else minsup
            # print(minsup)

            k_items += 1

            # Generate candidate sets Ck (set  of candidate k-sequences) -
            # generate new candidates from the last "best" candidates filtered
            # by minimum support
            items = np.unique( list(set(self.freq_patterns[k_items - 2].keys())))
            # originial
            # candidates = list(product(items, repeat=k_items))
            candidates = list(product(items, repeat=k_items)) if k_items <3 else self.filter_items_in_candidates(k_items,items)
            print("candidates: ")
            print(len(candidates))

            # candidate pruning - eliminates candidates who are not potentially
            # frequent (using support as threshold))
            self.item_suppocrt_dict = self._support(candidates)
            self.freq_patterns.append(self.filter_candidates(q, minsup_threshold,k_items))
            # self.freq_patterns.append(self._support(candidates, minsup))
            if k_items >2:
                print(self.freq_patterns[k_items-1])

            self._print_status(k_items, candidates)
        return self.freq_patterns

if __name__ == "__main__":

    def is_action_exist_in_pattern(actions_list, pattern):
        return tuple([i in set(pattern) for i in actions_list])

    con = psql.PostgressCon()
    con.connect()
    df = con.execute_query_with_headers("""select  game_id, action, team, play_number, player_name, 
                                        case when action in ('Canasta de 3', 'Canasta de 2','Canasta de 1','FIN DE PERIODO') then 1 else 0 end ind_canasta_2
                                        from stg.all_data_info
                                        where year = 2003
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
    print(len(df_lst))


    # Creating list of list for the apriori algo
    app_list = [i['pattern'] for i in df_lst]
    # if len(i['pattern']) <= 20]

    # patern_transaction_length = pd.Series([len(i) for i in app_list])
    # shorter_patern_transaction_length = pd.Series([len(i) for i in app_list  ])
    # patern_transaction_length.hist(bins =20)

    # Analyze
    t0 = time.time()
    result = GSP(app_list).search(50, 0.5, 0.99)
    print(result)
    t1 = time.time()
    print("DONE PREPROCESSING IN ", round((t1 - t0)), " SEC")

    # Result per epsilon [0.02,0.05,0.1]
    # dict = {'epsilon2':{
    #                     'run1':
    #                         {'candidates': 23
    #                          ,'filtered_candidates': 20},
    #                     'run2':
    #                         {'candidates': 400
    #                          ,'filtered_candidates': 38},
    #                     'run3':
    #                         {'candidates': 4096
    #                          ,'filtered_candidates': 33},
    #                     'run4':
    #                         {'candidates': 50625
    #                          ,'filtered_candidates': 4},
    #                     'run5':
    #                         {'candidates': 7778
    #                          ,'filtered_candidates': 0}
    #
    #                         },
    #         'epsilon5':{
    #                     'run1':
    #                         {'candidates': 23
    #                             , 'filtered_candidates': 17},
    #                     'run2':
    #                         {'candidates': 289
    #                             , 'filtered_candidates': 17},
    #                     'run3':
    #                         {'candidates': 1728
    #                             , 'filtered_candidates': 2},
    #                     'run4':
    #                         {'candidates': 81
    #                             , 'filtered_candidates': 0},
    #                     'run5':
    #                         {'candidates': 0
    #                             , 'filtered_candidates': 0}
    #
    #         }
    #         ,'epsilon10':{
    #
    #                     'run1':
    #                         {'candidates': 23
    #                             , 'filtered_candidates': 13},
    #                     'run2':
    #                         {'candidates': 169
    #                             , 'filtered_candidates': 6},
    #                     'run3':
    #                         {'candidates': 512
    #                             , 'filtered_candidates': 0},
    #                     'run4':
    #                         {'candidates': 0
    #                             , 'filtered_candidates': 0},
    #                     'run5':
    #                         {'candidates': 0
    #                             , 'filtered_candidates': 0}
    #         }
    # }
    #
    # data_cand = pd.DataFrame([
    #     ('run1', 'epsilon2', 23),
    #     ('run1', 'epsilon5', 23),
    #     ('run1', 'epsilon10', 23),
    #     ('run2', 'epsilon2', 400),
    #     ('run2', 'epsilon5', 289),
    #     ('run2', 'epsilon10', 169),
    #     ('run3', 'epsilon2', 4096),
    #     ('run3', 'epsilon5', 1728),
    #     ('run3', 'epsilon10', 512),
    #     ('run4', 'epsilon2', 50625),
    #     ('run4', 'epsilon5', 81),
    #     ('run4', 'epsilon10', 0),
    #     ('run5', 'epsilon2', 7778),
    #     ('run5', 'epsilon5', 0),
    #     ('run5', 'epsilon10', 0),
    # ],
    #     columns=['iteration#', 'Epsilon', 'Candidates']
    # )
    #
    # data_cand = data_cand.set_index(['iteration#', 'Epsilon']).sort_values('Candidates', ascending = False)
    #
    # temp = data_cand.unstack().plot(kind='bar',fontsize=20)
    # temp.legend(fontsize=20)
    # for p in temp.patches:
    #     temp.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=17)
    #
    # data_filter_cand = pd.DataFrame([
    #     ('run1', 'epsilon2', 20),
    #     ('run1', 'epsilon5', 17),
    #     ('run1', 'epsilon10', 13),
    #     ('run2', 'epsilon2', 38),
    #     ('run2', 'epsilon5', 17),
    #     ('run2', 'epsilon10', 6),
    #     ('run3', 'epsilon2', 33),
    #     ('run3', 'epsilon5', 2),
    #     ('run3', 'epsilon10', 0),
    #     ('run4', 'epsilon2', 4),
    #     ('run4', 'epsilon5', 0),
    #     ('run4', 'epsilon10', 0),
    #     ('run5', 'epsilon2', 0),
    #     ('run5', 'epsilon5', 0),
    #     ('run5', 'epsilon10', 0),
    # ],
    #     columns=['iteration#', 'Epsilon', 'Filtered_Candidates']
    # )
    #
    # data_filter_cand = data_filter_cand.set_index(['iteration#', 'Epsilon']).sort_values('Filtered_Candidates', ascending=False)
    #
    # data_filter_cand_plt = data_filter_cand.unstack().plot(kind='bar',fontsize=20)
    # data_filter_cand_plt.legend(fontsize=20)
    # for p in data_filter_cand_plt.patches:
    #     data_filter_cand_plt.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=17)