import utils.postgres_config as psql
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, learning_curve, KFold, GroupKFold, GridSearchCV, LeaveOneOut, PredefinedSplit
import pandas as pd
from random import shuffle, seed
import numpy as np
from dask_ml.wrappers import ParallelPostFit
from timeit import default_timer as timer
import time
from multiprocessing import Pool, freeze_support
import pickle
from joblib import dump, load
import xgboost as xgb
import sys
from utils.constant_class import constant

class model():

    def __init__(self, x_train, x_val, y_train, y_val ):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val


        validation_indices = np.zeros(y_train.shape[0])
        validation_indices[:round(3 / 4 * y_train.shape[0])] = -1
        self.validation_split = PredefinedSplit(validation_indices)

    def optimize_parameters(self, pipe, parameters):

        modelGs = GridSearchCV(pipe
                                   , parameters
                                   , cv = self.validation_split
                                   , refit=False
                                   , error_score='raise'
                                   , scoring=['f1_micro'])
        modelGs.fit(self.x_train, self.y_train)
        print("here1: ",np.argmax(modelGs.cv_results_['rank_test_f1_micro']))
        params = modelGs.cv_results_['params'][np.argmax(modelGs.cv_results_['rank_test_f1_micro'])]
        return params

    def train_knn(self):
        print("KNN START RUNNING")
        scalar = StandardScaler()
        knn = ParallelPostFit(KNeighborsClassifier())
        knnPipiLineDef = Pipeline([
            ('standartization', scalar),
            ('model', knn)
        ])

        t0 = time.time()
        parameters = self.optimize_parameters(knnPipiLineDef, {'model__estimator__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]})
        t1 = time.time()
        print("DONE CHOOSING PARAMS FOR KNN IN: ", round((t1 - t0)), " SEC")

        t0 = time.time()

        knnPipiLineDef.fit(self.x_train,self.y_train)
        self.knnPipiLineDef = knnPipiLineDef
        t1 = time.time()
        print("DONE TRAINING KNN IN: ", round((t1 - t0)), " SEC")
        return knnPipiLineDef

    def train_rnf(self):
        print("RNF START RUNNING")
        rnf = ParallelPostFit(RandomForestClassifier())
        rnfPipiLineDef = Pipeline([
            ('model', rnf)
        ])
        param = {'model__estimator__n_estimators': [20, 100, 500, 400]
                 , 'model__estimator__max_depth': [2, 10, 30]
                 , 'model__estimator__min_samples_split': [2, 5, 10]
                 , 'model__estimator__min_samples_leaf': [2,3,5]
                 , 'model__estimator__max_features': ['auto']
                 # , 'model__estimator__n_jobs': [-4]
                 , 'model__estimator__warm_start': [True]
        }

        t0 = time.time()
        parameters = self.optimize_parameters(rnfPipiLineDef, param)
        t1 = time.time()
        print("DONE CHOOSING PARAMS FOR RNF IN: ", round((t1 - t0) ), " SEC")

        t0 = time.time()
        rnfPipiLineDef.set_params(**parameters)
        setattr(self, "rnfPipiLineDef",rnfPipiLineDef.fit(self.x_train,self.y_train))
        t1 = time.time()
        print("DONE TRAINING RNF IN: ", round((t1 - t0) ), " SEC")
        return rnfPipiLineDef

    def train_svc(self):
        print("SVC START RUNNING")
        scalar = StandardScaler()
        svc = ParallelPostFit(SVC())
        svcPipiLineDef = Pipeline([
            ('standartization', scalar),
            ('model', svc)
        ])
        param = {'model__estimator__C': [0.1, 0.5, 1,3,5,10]
            , 'model__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            , 'model__estimator__shrinking': [True]
            , 'model__estimator__class_weight': ['balanced']
            , 'model__estimator__max_iter': [1000]
                 }

        t0 = time.time()
        parameters = self.optimize_parameters(svcPipiLineDef, param)
        t1 = time.time()
        print("DONE CHOOSING PARAMS FOR SVC IN: ", round((t1 - t0)), " SEC")

        t0 = time.time()
        svcPipiLineDef.set_params(**parameters)
        setattr(self, "svcPipiLineDef",svcPipiLineDef.fit(self.x_train,self.y_train))
        t1 = time.time()
        print("DONE TRAINING SVC IN: ", round((t1 - t0) ), " SEC")
        return svcPipiLineDef

    def train_adaboost(self):
        print("ADA START RUNNING")
        ada = ParallelPostFit(AdaBoostClassifier())
        adaPipiLineDef = Pipeline([
            ('model', ada)
        ])

        param = {'model__estimator__base_estimator': [DecisionTreeClassifier(max_depth=2)]
            , 'model__estimator__n_estimators': [5, 101, 501]
            , 'model__estimator__learning_rate': [0.1, 1, 10]
            , 'model__estimator__algorithm': ['SAMME']
                 }

        t0 = time.time()
        parameters = self.optimize_parameters(adaPipiLineDef, param)
        t1 = time.time()
        print("DONE CHOOSING PARAMS FOR ADA IN: ", round((t1 - t0)), " SEC")

        t0 = time.time()
        adaPipiLineDef.set_params(**parameters)
        setattr(self, "adaPipiLineDef", adaPipiLineDef.fit(self.x_train, self.y_train))
        t1 = time.time()
        print("DONE TRAINING ADA IN: ", round((t1 - t0)), " SEC")
        return adaPipiLineDef

    def train_dt(self):
        print("DT START RUNNING")
        dt = ParallelPostFit(DecisionTreeClassifier())
        dtPipiLineDef = Pipeline([
            ('model', dt)
        ])
        param = {'model__estimator__criterion': ['gini', 'entropy']
                 , 'model__estimator__max_depth': [2, 10, 30]
                 , 'model__estimator__min_samples_split': [2, 5, 10]
                 , 'model__estimator__min_samples_leaf': [2,3,5]
                 , 'model__estimator__max_features': ['auto']
                 }

        t0 = time.time()
        parameters = self.optimize_parameters(dtPipiLineDef, param)
        dtPipiLineDef.set_params(**parameters)
        t1 = time.time()
        print("DONE CHOOSING PARAMS FOR DT IN: ", round((t1 - t0)/60), " SEC")

        t0 = time.time()
        dtPipiLineDef.fit(self.x_train,self.y_train)
        self.dtPipiLineDef = dtPipiLineDef
        t1 = time.time()
        print("DONE TRAINING DT IN: ", round((t1 - t0) / 60), " SEC")
        return dtPipiLineDef

    def train_gradient_boost(self):
        print("GRA START RUNNING")
        gra = ParallelPostFit(GradientBoostingClassifier())
        graPipiLineDef = Pipeline([
            ('model', gra)
        ])

        param = {'model__estimator__n_estimators': [20, 100, 500, 400]
                 , 'model__estimator__max_depth': [2, 10, 30]
                 , 'model__estimator__min_samples_split': [2, 5, 10]
                 , 'model__estimator__min_samples_leaf': [2,3,5]
                 }

        t0 = time.time()
        parameters = self.optimize_parameters(graPipiLineDef, param)
        t1 = time.time()
        print("DONE CHOOSING PARAMS FOR GRA IN: ", round((t1 - t0)), " SEC")

        t0 = time.time()
        graPipiLineDef.set_params(**parameters)
        setattr(self, "graPipiLineDef", graPipiLineDef.fit(self.x_train, self.y_train))
        t1 = time.time()
        print("DONE TRAINING GRA IN: ", round((t1 - t0)), " SEC")
        return graPipiLineDef

    def train_simple_nn(self):
        pass

    def train_xgboost(self):
        print("XG START RUNNING")
        xgb_model = xgb.XGBClassifier(objective=" multi:softmax", random_state=42, num_class = len(self.y_train.unique()))
        le = LabelEncoder()
        self.y_train = pd.Series(le.fit_transform(self.y_train))
        t0 = time.time()
        if any(self.y_train.unique() < 0):
            self.y_train = self.y_train - min(self.y_train)
        setattr(self, "xgb_model", xgb_model.fit(self.x_train, self.y_train))

        t1 = time.time()
        print("DONE TRAINING xgb_model IN: ", round((t1 - t0)), " SEC")
        return xgb_model

    def run_train(self, func):
        return [func, getattr(self, func)()]
 # mod = getattr(self, func)()
 #        return [func,mod]

############################################################################################################################
# KNN

# OLOC MAIN
if __name__ == '__main__':

    t0 = time.time()

    q_for_tbl = constant.query_ml_oloc

    dbConn_obj = psql.PostgressCon()
    df = dbConn_obj.execute_query_with_headers(q_for_tbl)
    df = pd.DataFrame(data=df[0], columns=df[1])

    # Drop infinite an nan
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(0)

    # Handle dtypes
    df.team = df.team.astype('string')
    df.home_away = df.home_away.astype('string')
    df.q_minutes = df.q_minutes.astype('int')
    df.quarter = df.quarter.astype('string')
    df.sub_after_shot_by_team = df.sub_after_shot_by_team.astype('string')
    df.sub_after_miss_by_team = df.sub_after_miss_by_team.astype('string')
    df.sub_after_foul = df.sub_after_foul.astype('string')
    df.group_cv = df.group_cv.astype('string')
    df.cluster = df.cluster.astype('string')

    # Split train
    seed(42)

    shuffle_ind = list(range(0, df.shape[0] - 1))
    shuffle(shuffle_ind)
    df = df.iloc[shuffle_ind,].reset_index(drop=True)

    train = df.iloc[0:round(0.9 * df.shape[0]), :]
    test = df.iloc[round(0.9 * df.shape[0])::, :]

    # Handle categorical
    enc_team = pd.get_dummies(train.team)
    enc_ha = pd.get_dummies(train.home_away)
    enc_q_minutes = pd.get_dummies(train.q_minutes)
    enc_quarter = pd.get_dummies(train.quarter)
    enc_sub_after_shot_by_team = pd.get_dummies(train.sub_after_shot_by_team)
    enc_sub_after_miss_by_team = pd.get_dummies(train.sub_after_miss_by_team)
    enc_sub_after_foul = pd.get_dummies(train.sub_after_foul)
    enc_group_cv = pd.get_dummies(train.group_cv)
    enc_cluster = pd.get_dummies(train.cluster)

    train = pd.concat([train, enc_team], axis=1)
    train = pd.concat([train, enc_ha], axis=1)
    train = pd.concat([train, enc_quarter], axis=1)
    train = pd.concat([train, enc_sub_after_shot_by_team], axis=1)
    train = pd.concat([train, enc_sub_after_miss_by_team], axis=1)
    train = pd.concat([train, enc_sub_after_foul], axis=1)
    train = pd.concat([train, enc_group_cv], axis=1)
    train = pd.concat([train, enc_cluster], axis=1)

    # Train vs val
    x_col = ['score_dif', 'shot_score', 'shot_miss', 'foul_made',
             'foul_gain', 'cv_score_dif', 'away_team', 'home_team', 'sub_after_shot_0',
             'sub_after_shot_1', 'sub_after_miss_0', 'sub_after_miss_1',
             'sub_after_foul_0', 'sub_after_foul_1', 'High', 'Low', 'Mid',
             'Very high', 'Very low', 'high_team', 'low_team', 'med_team']
    y_col = ['oloc']
    x_train, x_val, y_train, y_val = train_test_split(train.loc[:, x_col],
                                                      train.loc[:, y_col], train_size=0.8, random_state=42)

    t1 = time.time()
    print("DONE PREPROCESSING IN ", round((t1 - t0)), " SEC")
    time.sleep(1)

    y_target = 'oloc'
    temp = model(x_train=x_train, x_val = x_val, y_train = y_train.loc[:,y_target], y_val = y_val.loc[:,y_target])
    # temp.run_train('train_xgboost')

    lst_of_model = sys.argv[1].split(',')
    pool = Pool(7)
    res = pool.map(temp.run_train ,lst_of_model)

    # train_gradient_boost
    pool.close()
    pool.join()

    print(res)
    path = r'C:\Users\yanir\PycharmProjects\masterThesis\ml_models'
    for model_lst in res:
        dump(model_lst[1], path+"\\"+model_lst[0]+'_' + y_target +'.joblib')

#
# # ALL ACTION MAIN
# if __name__ == '__main__':
#
#     t0 = time.time()
#
#     q_for_tbl = constant.query_ml_actions_value
#
#     dbConn_obj = psql.PostgressCon()
#     df = dbConn_obj.execute_query_with_headers(q_for_tbl)
#     df = pd.DataFrame(data=df[0], columns=df[1])
#
#     # Handle dtypes
#     df.team = df.team.astype('string')
#     df.home_away = df.home_away.astype('string')
#
#     # Split train
#     seed(42)
#
#     shuffle_ind = list(range(0, df.shape[0] - 1))
#     shuffle(shuffle_ind)
#     df = df.iloc[shuffle_ind,].reset_index(drop=True)
#
#     train = df.iloc[0:round(0.9 * df.shape[0]), :]
#     test = df.iloc[round(0.9 * df.shape[0])::, :]
#
#     # Handle categorical
#     enc_team = pd.get_dummies(train.team)
#     enc_ha = pd.get_dummies(train.home_away)
#
#     train = pd.concat([train, enc_team], axis=1)
#     train = pd.concat([train, enc_ha], axis=1)
#     train.drop(columns=['home_away', 'team'], inplace=True)
#
#     # Train vs val
#     x_col = ['year', 'time_marker', 'five_on_court',
#              'ALI', 'BAR', 'BAS', 'BRE', 'CLA', 'EST',
#              'FUE', 'GIR', 'JOV', 'LLE', 'MAD', 'MAN',
#              'MUR', 'PAM', 'RON', 'SEV', 'TNF', 'VAL',
#              'away_team', 'home_team']
#     y_col = ['shot_score', 'shot_miss', 'drb', 'orb', 'stl', 'turn',
#              'foul_made', 'foul_gain', 'quarter', 'action_status_dir',
#              'action_status_sum', 'action_status_canasta', 'is_good__dir_actions',
#              'is_good_sum_actions']
#     x_train, x_val, y_train, y_val = train_test_split(train.loc[:, x_col],
#                                                       train.loc[:, y_col], train_size=0.8, random_state=42)
#
#     print(sys.argv)
#     # y_target = sys.argv[-1]
#     y_target = 'action_status_sum'
#     temp = model(x_train=x_train, x_val = x_val, y_train = y_train.loc[:,y_target], y_val = y_val.loc[:,y_target])
#
#     t1 = time.time()
#     print("DONE PREPROCESSING IN ", round((t1 - t0)), " SEC")
#     time.sleep(1)
#
#     pool = Pool(7)
#     res = pool.map(temp.run_train ,['train_xgboost'])
#                    # ['train_dt', 'train_knn', 'train_rnf', 'train_svc', 'train_xgboost', 'train_adaboost'])
#     # train_gradient_boost
#     pool.close()
#     pool.join()
#
#     print(res)
#     path = r'C:\Users\yanir\PycharmProjects\masterThesis\ml_models'
#     for model_lst in res:
#         dump(model_lst[1], path+"\\"+model_lst[0]+'_' + y_target +'.joblib')
