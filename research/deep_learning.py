
import pandas as pd
from random import shuffle, seed
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import itertools
from joblib import dump, load
import utils.postgres_config as psql
from utils.constant_class import constant
import time
from tensorflow import config
config.list_physical_devices('GPU')

from sklearn.neural_network import MLPClassifier

import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split

class NnModel():
    
    def __init__(self,train = None, test = None, val = None):
        self.train = train
        self.test = test
        self.val = val
        self.feature_layer_inputs = {}
        self.feature_layer_lst =[]
        self.all_inputs = []
        self.encoded_features = []
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None

    def setTrain(self, df):
        self.train = df

    def setTest(self, df):
        self.test = df

    def setVal(self, df):
        self.val = df

    def setTrainDs(self, df):
        self.train_ds = df

    def setTestDs(self, df):
        self.test_ds = df

    def setValDs(self, df):
        self.val_ds = df

    def demo(self, feature_column):
        self.feature_layer_lst.append(feature_column)

    def one_hot_encoder(self, key,voc_lst):
        cat_feature = feature_column.categorical_column_with_vocabulary_list(key, list(voc_lst))
        cat_feature_one_hot = feature_column.indicator_column(cat_feature)
        self.demo(cat_feature_one_hot)

    def encode_numerical_train(self, col, train = True):
        feature_col = feature_column.numeric_column(col)
        self.demo(feature_col)

    def model(self):

        feature_layer = tf.keras.layers.DenseFeatures(self.feature_layer_lst)
        # feature_layer_outputs = feature_layer(self.feature_layer_inputs)

        model = tf.keras.Sequential([
            feature_layer,
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(name="BN1"),
            layers.Dropout(.9, name='d1'),
            layers.Dense(128, activation='relu'),
            layers.Dropout(.8, name='d2'),
            layers.Dense(128, activation='relu'),
            layers.Dense(5, activation = 'softmax', kernel_regularizer=l2(10.00), name = 'Output')
        ])

        opt = SGD(learning_rate=0.00001, momentum=0.7, decay=0.005, nesterov=False)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

    def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
      dataframe = dataframe.copy()
      labels =  np.asarray(dataframe.pop('oloc')).astype('int32')
      ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
      if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
      ds = ds.batch(batch_size)
      return ds

if __name__ == "__main__":
    q_for_tbl = constant.query_ml_oloc

    dbConn_obj = psql.PostgressCon()
    df = dbConn_obj.execute_query_with_headers(q_for_tbl)
    df = pd.DataFrame(data=df[0], columns=df[1])

    df.time_marker = df.time_marker.astype("int32")
    df.cv_score_dif = df.cv_score_dif.astype("float32")
    df.abs_cv_score_dif = df.abs_cv_score_dif.astype("float32")
    df.score_dif = df.score_dif.astype("int32")
    df.shot_score = df.shot_score.astype("int32")
    df.shot_miss = df.shot_miss.astype("int32")
    df.foul_made = df.foul_made.astype("int32")
    df.foul_gain = df.foul_gain.astype("int32")
    df.q_minutes = df.q_minutes.astype("int32")

    df.oloc = df.oloc.astype("string")

    numeric_list = ['score_dif', 'shot_score', 'shot_miss', 'foul_made', 'foul_gain', 'q_minutes']
    categorical_list = ['quarter', 'home_away', 'team', 'sub_after_shot_by_team', 'sub_after_miss_by_team'
        , 'sub_after_foul', 'cv_score_dif', 'abs_cv_score_dif', 'group_cv', 'cluster']


    x_col = ['quarter', 'q_minutes', 'team','score_dif', 'shot_score'
        , 'shot_miss', 'foul_made', 'foul_gain', 'home_away'
        , 'sub_after_shot_by_team', 'sub_after_miss_by_team'
        , 'sub_after_foul', 'cv_score_dif', 'abs_cv_score_dif'
        , 'group_cv', 'cluster'
             ]
    y_col = ['oloc']


    # Split train
    seed(42)
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    mod = NnModel(train, test, val )

    ################################

    batch_size = 5  # A small batch sized is used for demonstration purposes
    train_ds = mod.df_to_dataset(train, batch_size=batch_size)
    val_ds = mod.df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = mod.df_to_dataset(test, shuffle=False, batch_size=batch_size)

    mod.train_ds = train_ds
    mod.val_ds = val_ds
    mod.test_ds = test_ds

    for header in numeric_list:
        mod.encode_numerical_train(header)

    for header in categorical_list:
        mod.encode_categorical_train(header)

    [(train_features, label_batch)] = train_ds.take(1)
    print('Every feature:', list(train_features.keys()))
    print('A batch of ages:', train_features['quarter'])
    print('A batch of targets:', label_batch)


    # Handle dtypes
    # mod.one_hot_encoder('team', df.team.unique())
    # mod.one_hot_encoder('home_away', df.home_away.unique())
    # mod.one_hot_encoder('quarter', df.quarter.unique())
    # mod.one_hot_encoder('sub_after_shot_by_team', df.sub_after_shot_by_team.unique())
    # mod.one_hot_encoder('sub_after_miss_by_team', df.sub_after_miss_by_team.unique())
    # mod.one_hot_encoder('sub_after_foul', df.sub_after_foul.unique())
    # mod.one_hot_encoder('group_cv', df.group_cv.unique())
    # mod.one_hot_encoder('cluster', df.cluster.unique())
    #
    # mod.encode_numerical('q_minutes')
    # mod.encode_numerical('score_dif')
    # mod.encode_numerical('shot_score')
    # mod.encode_numerical('shot_miss')
    # mod.encode_numerical('foul_made')
    # mod.encode_numerical('foul_gain')
    # mod.encode_numerical('cv_score_dif')
    # mod.encode_numerical('abs_cv_score_dif')

    my_nn_model = mod.model()

    my_nn_model.fit(train_ds, validation_data=val_ds, epochs=10)

    my_nn_model.summary()
    loss, accuracy = my_nn_model.evaluate(test_ds)
    print("Accuracy", accuracy)

