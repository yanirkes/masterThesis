import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import utils.postgres_config as psql
from utils.constant_class import constant
from tensorflow.keras.utils import plot_model
from tensorflow import config
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from joblib import dump, load
from random import shuffle, seed
import time
import eli5
from eli5.sklearn import PermutationImportance
config.list_physical_devices('GPU')
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


q_for_tbl = constant.query_ml_oloc

dbConn_obj = psql.PostgressCon()
df = dbConn_obj.execute_query_with_headers(q_for_tbl)
df = pd.DataFrame(data=df[0], columns=df[1])

df.time_marker = df.time_marker.astype("int32")
df.cv_score_dif = df.cv_score_dif.astype("float32")
df.abs_cv_score_dif = df.abs_cv_score_dif.astype("float32")
df.score_dif = df.score_dif.astype("int32")
df.q_minutes = df.q_minutes.astype("int32")

df.shot_score = df.shot_score.astype("string")
df.shot_miss = df.shot_miss.astype("string")
df.foul_made = df.foul_made.astype("string")
df.foul_gain = df.foul_gain.astype("string")
df.sub_after_shot_by_team = df.sub_after_shot_by_team.astype("string")
df.sub_after_miss_by_team = df.sub_after_miss_by_team.astype("string")
df.sub_after_foul = df.sub_after_foul.astype("string")
df.group_cv = df.group_cv.astype("string")
df.cluster = df.cluster.astype("string")

df.oloc = df.oloc.astype("int32")
df.oloc = df.oloc-1


# Handle categorical
# enc_team = pd.get_dummies(train.team)
# enc_ha = pd.get_dummies(train.home_away)
# enc_q_minutes = pd.get_dummies(train.q_minutes)
# enc_quarter = pd.get_dummies(train.quarter)
# enc_sub_after_shot_by_team = pd.get_dummies(train.sub_after_shot_by_team)
# enc_sub_after_miss_by_team = pd.get_dummies(train.sub_after_miss_by_team)
# enc_sub_after_foul = pd.get_dummies(train.sub_after_foul)
# enc_group_cv = pd.get_dummies(train.group_cv)
# enc_cluster = pd.get_dummies(train.cluster)
#
# train = pd.concat([train, enc_team], axis=1)
# train = pd.concat([train, enc_ha], axis=1)
# train = pd.concat([train, enc_quarter], axis=1)
# train = pd.concat([train, enc_sub_after_shot_by_team], axis=1)
# train = pd.concat([train, enc_sub_after_miss_by_team], axis=1)
# train = pd.concat([train, enc_sub_after_foul], axis=1)
# train = pd.concat([train, enc_group_cv], axis=1)
# train = pd.concat([train, enc_cluster], axis=1)



numeric_list = [
                # 'score_dif',
                'q_minutes',
                # 'cv_score_dif',
                # 'abs_cv_score_dif'
                ]
categorical_list = ['quarter',
                    'home_away',
                    'team',
                    # 'shot_score',
                    # 'shot_miss',
                    # 'foul_made',
                    # 'foul_gain',
                    # 'sub_after_shot_by_team',
                    # 'sub_after_miss_by_team',
                    # 'sub_after_foul',
                    'group_cv',
                    'cluster']


x_col = ['quarter',
         'q_minutes',
         'team',
         # 'score_dif',
         # 'shot_score',
         # 'shot_miss',
         # 'foul_made',
         # 'foul_gain',
         'home_away' ,
         # 'sub_after_shot_by_team',
         # 'sub_after_miss_by_team' ,
         # 'sub_after_foul',
         # 'cv_score_dif',
         # 'abs_cv_score_dif',
         'group_cv',
         'cluster'
         ]

y_col = ['oloc']


# Split train
seed(42)
train, test = train_test_split(df.loc[:,x_col+y_col], test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

def get_category_encoding_layer( name, dataset, dtype, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

def get_normalization_layer(dataset, col):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[col])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

@tf.autograph.experimental.do_not_convert
def encode_categorical_train( col, train):
    categorical_col = tf.keras.Input(shape=(1,), name=col, dtype='string')
    map_ = get_category_encoding_layer(name=col,dataset = train, dtype='string',max_tokens=5)
    encoded_categorical_col = map_(categorical_col)
    return categorical_col, encoded_categorical_col

@tf.autograph.experimental.do_not_convert
def encode_numerical_train( col, train):
    numeric_col = tf.keras.Input(shape=(1,), name=col)
    normalization_layer = get_normalization_layer(train, col)
    encoded_numeric_col = normalization_layer(numeric_col)
    return numeric_col, encoded_numeric_col

def run(run_dir, train_ds, val_ds, encoded_features, all_inputs,hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(train_ds, val_ds, encoded_features, all_inputs,hparams)
    tf.summary.scalar('accuracy', accuracy, step=1)


def train_test_model(train_ds, val_ds, encoded_features, all_inputs,hparams):

    all_features = tf.keras.layers.concatenate(encoded_features)

    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation="relu")(all_features)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation="relu")(x)
    x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)
    x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation="relu", kernel_regularizer=l2(10.00))(x)
    output = tf.keras.layers.Dense(5)(x)

    model = tf.keras.Model(all_inputs, output)

    opt = Adam(learning_rate = hparams[HP_LR]) if hparams[HP_OPTIMIZER] == 'adam' else SGD(learning_rate = hparams[HP_LR], momentum = hparams[HP_MOM])

    model.compile(
        optimizer=opt,
        loss='CategoricalCrossentropy',
        metrics=['accuracy'],
    )

    model.fit(train_ds, epochs=3,
              callbacks=[
                  tf.keras.callbacks.TensorBoard('ml_models/hparam_tuning'),  # log metrics
                  hp.KerasCallback('ml_models/hparam_tuning', hparams),  # log hparams
              ],
              )  # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(val_ds)
    return accuracy


def model(encoded_features, all_inputs):


    all_features = tf.keras.layers.concatenate(encoded_features)

    x = tf.keras.layers.Dense(200, activation="relu")(all_features)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    output = tf.keras.layers.Dense(5)(x)
    opt = SGD(learning_rate=1, momentum=0.9, decay=0.80, nesterov=False)

    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels =  np.asarray(dataframe.pop('oloc')).astype('int32')
  labels =  to_categorical(labels,  num_classes = 5)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


batch_size = 180 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []
encoded_features_cat = []

for header in numeric_list:
    print(header)
    a,b = encode_numerical_train(header, train_ds)
    all_inputs .append(a)
    encoded_features.append(b)

for header in categorical_list:
    print(header)
    a,b = encode_categorical_train(header, train_ds)
    all_inputs.append(a)
    encoded_features_cat.append(b)

temp = encoded_features+encoded_features_cat

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([200, 250, 500,1000]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.5, 0.7,0.9]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.1, 0.001,0.00001]))
HP_MOM = hp.HParam('momentum', hp.Discrete([0.1, 0.5,0.9]))

with tf.summary.create_file_writer('ml_models/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_LR, HP_MOM],
        metrics=[hp.Metric('accuracy', display_name='Accuracy')],
    )
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in HP_DROPOUT.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for lr in HP_LR.domain.values:
                for mom in HP_MOM.domain.values:
                    hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                        HP_LR: lr,
                        HP_MOM: mom
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('ml_models/hparam_tuning/' + run_name, train_ds, val_ds, encoded_features, all_inputs,hparams)
                    session_num += 1



my_nn_model = model(temp, all_inputs)

history = my_nn_model.fit(train_ds, validation_data=val_ds, epochs=11, callbacks=[callback])

my_nn_model.summary()
loss, accuracy = my_nn_model.evaluate(test_ds)
print("Accuracy", accuracy)


pd.DataFrame(history.history).plot()

temp_pred = my_nn_model.predict(val_ds)
temp_pred = temp_pred.argmax(axis=1)




# test_type_col = train['home_away']
# test_type_layer = get_category_encoding_layer(name='home_away', dataset=train_ds, dtype='string')
# test_type_layer(test_type_col)

