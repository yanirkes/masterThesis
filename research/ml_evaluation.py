from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, average_precision_score, \
    roc_auc_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, learning_curve, KFold, \
    GroupKFold, GridSearchCV, LeaveOneOut, PredefinedSplit
import xgboost as xgb
import pandas as pd
from random import shuffle, seed
import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import dump, load
import seaborn as sns
import utils.postgres_config as psql
from utils.constant_class import constant
import time


def evaluate_mul_class(y_pred, y_true, title=None):
    f1micro = f1_score(y_true, y_pred, average='micro')
    f1macro = f1_score(y_true, y_pred, average='macro')
    f1weighted = f1_score(y_true, y_pred, average='weighted')
    # prec = average_precision_score(y_true, y_pred, average='macro')
    # auc = roc_auc_score(y_true, y_pred, average='macro')
    print(title, "results are:\nf1micro: %s, f1macro: %s, f1weighted: %s" % (
        round(f1micro, 2), round(f1macro, 2), round(f1weighted, 2)))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


########################################################
###########       ALL ACTIONS MODEL ####################
########################################################

q_for_tbl = constant.query_ml_actions_value

dbConn_obj = psql.PostgressCon()
df = dbConn_obj.execute_query_with_headers(q_for_tbl)
df = pd.DataFrame(data=df[0], columns=df[1])

# Handle dtypes
df.team = df.team.astype('string')
df.home_away = df.home_away.astype('string')

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

train = pd.concat([train, enc_team], axis=1)
train = pd.concat([train, enc_ha], axis=1)
train.drop(columns=['home_away', 'team'], inplace=True)

# Train vs val
x_col = ['year', 'time_marker', 'five_on_court',
         'ALI', 'BAR', 'BAS', 'BRE', 'CLA', 'EST',
         'FUE', 'GIR', 'JOV', 'LLE', 'MAD', 'MAN',
         'MUR', 'PAM', 'RON', 'SEV', 'TNF', 'VAL',
         'away_team', 'home_team']
y_col = ['shot_score', 'shot_miss', 'drb', 'orb', 'stl', 'turn',
         'foul_made', 'foul_gain', 'quarter', 'action_status_dir',
         'action_status_sum', 'action_status_canasta', 'is_good__dir_actions',
         'is_good_sum_actions']
x_train, x_val, y_train, y_val = train_test_split(train.loc[:, x_col],
                                                  train.loc[:, y_col], train_size=0.8, random_state=42)

## SHOT SCORE
adaboost = load("ml_models/train_adaboost_shot_score.joblib")
dt = load("ml_models/train_dt_shot_score.joblib")
knn = load("ml_models/train_knn_shot_score.joblib")
rnf = load("ml_models/train_rnf_shot_score.joblib")
svc = load("ml_models/train_svc_shot_score.joblib")
xgboost = load("ml_models/train_xgboost_shot_score.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.shot_score, 'train_adaboost_shot_score')
evaluate_mul_class(dt_train, y_train.shot_score, 'dt_train_shot_score')
evaluate_mul_class(knn_train, y_train.shot_score, 'knn_train_shot_score')
evaluate_mul_class(rnf_train, y_train.shot_score, 'rnf_train_shot_score')
evaluate_mul_class(svc_train, y_train.shot_score, 'svc_train_shot_score')
evaluate_mul_class(xgboost_train, y_train.shot_score, 'xgboost_train_shot_score')

evaluate_mul_class(adaboost_val, y_val.shot_score, 'adaboost_val_shot_score')
evaluate_mul_class(dt_val, y_val.shot_score, 'dt_val_shot_score')
evaluate_mul_class(knn_val, y_val.shot_score, 'knn_val_shot_score')
evaluate_mul_class(rnf_val, y_val.shot_score, 'rnf_val_shot_score')
evaluate_mul_class(svc_val, y_val.shot_score, 'svc_val_shot_score')
evaluate_mul_class(xgboost_val, y_val.shot_score, 'xgboost_val_shot_score')

## SHOT MISS
adaboost = load("ml_models/train_adaboost_shot_miss.joblib").named_steps['model']
dt = load("ml_models/train_dt_shot_miss.joblib").named_steps['model']
knn = load("ml_models/train_knn_shot_miss.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_shot_miss.joblib").named_steps['model']
svc = load("ml_models/train_svc_shot_miss.joblib").named_steps['model']
xgboost = load("ml_models/train_xgboost_shot_miss.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.shot_miss, 'train_adaboost_shot_miss')
evaluate_mul_class(dt_train, y_train.shot_miss, 'dt_train_shot_miss')
evaluate_mul_class(knn_train, y_train.shot_miss, 'knn_train_shot_miss')
evaluate_mul_class(rnf_train, y_train.shot_miss, 'rnf_train_shot_miss')
evaluate_mul_class(svc_train, y_train.shot_miss, 'svc_train_shot_miss')
evaluate_mul_class(xgboost_train, y_train.shot_miss, 'xgboost_train_shot_miss')

evaluate_mul_class(adaboost_val, y_val.shot_miss, 'adaboost_val_shot_miss')
evaluate_mul_class(dt_val, y_val.shot_miss, 'dt_val_shot_miss')
evaluate_mul_class(knn_val, y_val.shot_miss, 'knn_val_shot_miss')
evaluate_mul_class(rnf_val, y_val.shot_miss, 'rnf_val_shot_miss')
evaluate_mul_class(svc_val, y_val.shot_miss, 'svc_val_shot_miss')
evaluate_mul_class(xgboost_val, y_val.shot_miss, 'xgboost_val_shot_miss')

## DRB
adaboost = load("ml_models/train_adaboost_drb.joblib").named_steps['model']
dt = load("ml_models/train_dt_drb.joblib").named_steps['model']
knn = load("ml_models/train_knn_drb.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_drb.joblib").named_steps['model']
svc = load("ml_models/train_svc_drb.joblib").named_steps['model']
xgboost = load("ml_models/train_xgboost_drb.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.drb, 'train_adaboost_drb')
evaluate_mul_class(dt_train, y_train.drb, 'dt_train_drb')
evaluate_mul_class(knn_train, y_train.drb, 'knn_train_drb')
evaluate_mul_class(rnf_train, y_train.drb, 'rnf_train_drb')
evaluate_mul_class(svc_train, y_train.drb, 'svc_train_drb')
evaluate_mul_class(xgboost_train, y_train.drb, 'xgboost_train_drb')
print('\n')

evaluate_mul_class(adaboost_val, y_val.drb, 'adaboost_val_drb')
evaluate_mul_class(dt_val, y_val.drb, 'dt_val_drb')
evaluate_mul_class(knn_val, y_val.drb, 'knn_val_drb')
evaluate_mul_class(rnf_val, y_val.drb, 'rnf_val_drb')
evaluate_mul_class(svc_val, y_val.drb, 'svc_val_drb')
evaluate_mul_class(xgboost_val, y_val.drb, 'xgboost_val_drb')
print('\n')

## ORB
adaboost = load("ml_models/train_adaboost_orb.joblib").named_steps['model']
dt = load("ml_models/train_dt_orb.joblib").named_steps['model']
knn = load("ml_models/train_knn_orb.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_orb.joblib").named_steps['model']
svc = load("ml_models/train_svc_orb.joblib").named_steps['model']

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.orb, 'train_adaboost_orb')
evaluate_mul_class(dt_train, y_train.orb, 'dt_train_orb')
evaluate_mul_class(knn_train, y_train.orb, 'knn_train_orb')
evaluate_mul_class(rnf_train, y_train.orb, 'rnf_train_orb')
evaluate_mul_class(svc_train, y_train.orb, 'svc_train_orb')
evaluate_mul_class(xgboost_train, y_train.orb, 'xgboost_train_orb')
print('\n')

evaluate_mul_class(adaboost_val, y_val.orb, 'adaboost_val_orb')
evaluate_mul_class(dt_val, y_val.orb, 'dt_val_orb')
evaluate_mul_class(knn_val, y_val.orb, 'knn_val_orb')
evaluate_mul_class(rnf_val, y_val.orb, 'rnf_val_orb')
evaluate_mul_class(svc_val, y_val.orb, 'svc_val_orb')
evaluate_mul_class(xgboost_val, y_val.orb, 'xgboost_val_orb')
print('\n')

## STL
adaboost = load("ml_models/train_adaboost_stl.joblib").named_steps['model']
dt = load("ml_models/train_dt_stl.joblib").named_steps['model']
knn = load("ml_models/train_knn_stl.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_stl.joblib").named_steps['model']
svc = load("ml_models/train_svc_stl.joblib").named_steps['model']
xgboost = load("ml_models/train_xgboost_stl.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.stl, 'train_adaboost_stl')
evaluate_mul_class(dt_train, y_train.stl, 'dt_train_stl')
evaluate_mul_class(knn_train, y_train.stl, 'knn_train_stl')
evaluate_mul_class(rnf_train, y_train.stl, 'rnf_train_stl')
evaluate_mul_class(svc_train, y_train.stl, 'svc_train_stl')
evaluate_mul_class(xgboost_train, y_train.stl, 'xgboost_train_stl')
print('\n')

evaluate_mul_class(adaboost_val, y_val.stl, 'adaboost_val_stl')
evaluate_mul_class(dt_val, y_val.stl, 'dt_val_stl')
evaluate_mul_class(knn_val, y_val.stl, 'knn_val_stl')
evaluate_mul_class(rnf_val, y_val.stl, 'rnf_val_stl')
evaluate_mul_class(svc_val, y_val.stl, 'svc_val_stl')
evaluate_mul_class(xgboost_val, y_val.stl, 'xgboost_val_stl')
print('\n')

## TURNOVER
adaboost = load("ml_models/train_adaboost_turn.joblib").named_steps['model']
dt = load("ml_models/train_dt_turn.joblib").named_steps['model']
knn = load("ml_models/train_knn_turn.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_turn.joblib").named_steps['model']
# svc = load("ml_models/train_svc_tun.joblib").named_steps['model']
xgboost = load("ml_models/train_xgboost_turn.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
# svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
# svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.turn, 'train_adaboost_turn')
evaluate_mul_class(dt_train, y_train.turn, 'dt_train_turn')
evaluate_mul_class(knn_train, y_train.turn, 'knn_train_turn')
evaluate_mul_class(rnf_train, y_train.turn, 'rnf_train_turn')
# evaluate_mul_class(svc_train, y_train.turn, 'svc_train_turn')
evaluate_mul_class(xgboost_train, y_train.turn, 'xgboost_train_turn')
print('\n')

evaluate_mul_class(adaboost_val, y_val.turn, 'adaboost_val_turn')
evaluate_mul_class(dt_val, y_val.turn, 'dt_val_turn')
evaluate_mul_class(knn_val, y_val.turn, 'knn_val_turn')
evaluate_mul_class(rnf_val, y_val.turn, 'rnf_val_turn')
# evaluate_mul_class(svc_val, y_val.turn, 'svc_val_turn')
evaluate_mul_class(xgboost_val, y_val.turn, 'xgboost_val_turn')
print('\n')

## FOUL GAIN
adaboost = load("ml_models/train_adaboost_foul_gain.joblib").named_steps['model']
dt = load("ml_models/train_dt_foul_gain.joblib").named_steps['model']
knn = load("ml_models/train_knn_foul_gain.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_foul_gain.joblib").named_steps['model']
# svc = load("ml_models/train_svc_foul_gain.joblib").named_steps['model']
xgboost = load("ml_models/train_xgboost_foul_gain.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
# svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
# svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.foul_gain, 'train_adaboost_foul_gain')
evaluate_mul_class(dt_train, y_train.foul_gain, 'dt_train_foul_gain')
evaluate_mul_class(knn_train, y_train.foul_gain, 'knn_train_foul_gain')
evaluate_mul_class(rnf_train, y_train.foul_gain, 'rnf_train_foul_gain')
# evaluate_mul_class(svc_train, y_train.foul_gain, 'svc_train_foul_gain')
evaluate_mul_class(xgboost_train, y_train.foul_gain, 'xgboost_train_foul_gain')
print('\n')

evaluate_mul_class(adaboost_val, y_val.foul_gain, 'adaboost_val_foul_gain')
evaluate_mul_class(dt_val, y_val.foul_gain, 'dt_val_foul_gain')
evaluate_mul_class(knn_val, y_val.foul_gain, 'knn_val_foul_gain')
evaluate_mul_class(rnf_val, y_val.foul_gain, 'rnf_val_foul_gain')
# evaluate_mul_class(svc_val, y_val.goul_gain, 'svc_val_goul_gain')
evaluate_mul_class(xgboost_val, y_val.foul_gain, 'xgboost_val_foul_gain')
print('\n')

## FOUL MADE
adaboost = load("ml_models/train_adaboost_foul_made.joblib").named_steps['model']
dt = load("ml_models/train_dt_foul_made.joblib").named_steps['model']
knn = load("ml_models/train_knn_foul_made.joblib").named_steps['model']
rnf = load("ml_models/train_rnf_foul_made.joblib").named_steps['model']
# svc = load("ml_models/train_svc_foul_made.joblib").named_steps['model']
xgboost = load("ml_models/train_xgboost_foul_made.joblib")

# TRAIN
adaboost_train = adaboost.predict(x_train)
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
# svc_train = svc.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
adaboost_val = adaboost.predict(x_val)
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
# svc_val = svc.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(adaboost_train, y_train.foul_made, 'train_adaboost_foul_made')
evaluate_mul_class(dt_train, y_train.foul_made, 'dt_train_foul_made')
evaluate_mul_class(knn_train, y_train.foul_made, 'knn_train_foul_made')
evaluate_mul_class(rnf_train, y_train.foul_made, 'rnf_train_foul_made')
# evaluate_mul_class(svc_train, y_train.foul_made, 'svc_train_foul_made')
evaluate_mul_class(xgboost_train, y_train.foul_made, 'xgboost_train_foul_made')
print('\n')

evaluate_mul_class(adaboost_val, y_val.foul_made, 'adaboost_val_foul_made')
evaluate_mul_class(dt_val, y_val.foul_made, 'dt_val_foul_made')
evaluate_mul_class(knn_val, y_val.foul_made, 'knn_val_foul_made')
evaluate_mul_class(rnf_val, y_val.foul_made, 'rnf_val_foul_made')
# evaluate_mul_class(svc_val, y_val.foul_made, 'svc_val_foul_made')
evaluate_mul_class(xgboost_val, y_val.foul_made, 'xgboost_val_foul_made')

xgboost_aot_score = load("ml_models/train_xgboost_shot_score.joblib")
xgboost_orn = load("ml_models/train_xgboost_orb.joblib")
xgboost_stl = load("ml_models/train_xgboost_stl.joblib")
xgboost_turn = load("ml_models/train_xgboost_turn.joblib")

# XGBOOST

train_xgboost_action_status_canasta = load("ml_models/train_xgboost_action_status_canasta.joblib")
train_xgboost_action_status_sum = load("ml_models/train_xgboost_action_status_sum.joblib")
train_xgboost_shot_score = load("ml_models/train_xgboost_shot_score.joblib")
train_xgboost_shot_miss = load("ml_models/train_xgboost_shot_miss.joblib")
train_xgboost_drb = load("ml_models/train_xgboost_drb.joblib")
train_xgboost_stl = load("ml_models/train_xgboost_stl.joblib")
train_xgboost_turn = load("ml_models/train_xgboost_turn.joblib")
train_xgboost_foul_gain = load("ml_models/train_xgboost_foul_gain.joblib")
train_xgboost_foul_made = load("ml_models/train_xgboost_foul_made.joblib")

le = LabelEncoder()
y_trainaction_status_canasta = pd.Series(le.fit_transform(y_train.action_status_canasta))
y_trainaction_status_sum = pd.Series(le.fit_transform(y_train.action_status_sum))
y_valaction_status_canasta = pd.Series(le.fit_transform(y_val.action_status_canasta))
y_valaction_status_sum = pd.Series(le.fit_transform(y_val.action_status_sum))

train_xgboost_shot_score_train = train_xgboost_shot_score.predict(x_train)
train_xgboost_shot_miss_train = train_xgboost_shot_miss.predict(x_train)
train_xgboost_drb_train = train_xgboost_drb.predict(x_train)
train_xgboost_stl_train = train_xgboost_stl.predict(x_train)
train_xgboost_turn_train = train_xgboost_turn.predict(x_train)
train_xgboost_foul_gain_train = train_xgboost_foul_gain.predict(x_train)
train_xgboost_foul_made_train = train_xgboost_foul_made.predict(x_train)

train_xgboost_shot_score_val = train_xgboost_shot_score.predict(x_val)
train_xgboost_shot_miss_val = train_xgboost_shot_miss.predict(x_val)
train_xgboost_drb_val = train_xgboost_drb.predict(x_val)
train_xgboost_stl_val = train_xgboost_stl.predict(x_val)
train_xgboost_turn_val = train_xgboost_turn.predict(x_val)
train_xgboost_foul_gain_val = train_xgboost_foul_gain.predict(x_val)
train_xgboost_foul_made_val = train_xgboost_foul_made.predict(x_val)

xgboost1_tr_pred = train_xgboost_action_status_canasta.predict(x_train)
xgboost2_tr_pred = train_xgboost_action_status_sum.predict(x_train)
xgboost1_val_pred = train_xgboost_action_status_canasta.predict(x_val)
xgboost2_val_pred = train_xgboost_action_status_sum.predict(x_val)

evaluate_mul_class(train_xgboost_shot_score_train, y_train.shot_score, 'xgboost_train_shot_score')
evaluate_mul_class(train_xgboost_shot_score_val, y_val.shot_score, 'xgboost_val_shot_score')
evaluate_mul_class(train_xgboost_shot_miss_train, y_train.shot_miss, 'xgboost_train_shot_miss')
evaluate_mul_class(train_xgboost_shot_miss_val, y_val.shot_miss, 'xgboost_val_shot_miss')
evaluate_mul_class(train_xgboost_drb_train, y_train.drb, 'xgboost_train_drb')
evaluate_mul_class(train_xgboost_drb_val, y_val.drb, 'xgboost_val_drb')
evaluate_mul_class(train_xgboost_stl_train, y_train.stl, 'xgboost_train_stl')
evaluate_mul_class(train_xgboost_stl_val, y_val.stl, 'xgboost_val_stl')
evaluate_mul_class(train_xgboost_turn_train, y_train.turn, 'xgboost_train_turn')
evaluate_mul_class(train_xgboost_turn_val, y_val.turn, 'xgboost_val_turn')
evaluate_mul_class(train_xgboost_foul_gain_train, y_train.foul_gain, 'xgboost_train_foul_gain')
evaluate_mul_class(train_xgboost_foul_gain_val, y_val.foul_gain, 'xgboost_val_foul_gain')
evaluate_mul_class(train_xgboost_foul_made_train, y_train.foul_made, 'xgboost_train_foul_made')
evaluate_mul_class(train_xgboost_foul_made_val, y_val.foul_made, 'xgboost_val_foul_made')
evaluate_mul_class(xgboost1_tr_pred, y_trainaction_status_canasta, 'xgboost1_val_foul_made')
evaluate_mul_class(xgboost2_tr_pred, y_trainaction_status_sum, 'xgboost2_val_foul_made')
evaluate_mul_class(xgboost1_val_pred, y_valaction_status_canasta, 'xgboost1_val_foul_made')
evaluate_mul_class(xgboost2_val_pred, y_valaction_status_sum, 'xgboost2_val_foul_made')

cm_shot_shotscore = confusion_matrix(xgboost1_val_pred, y_valaction_status_sum)
plot_confusion_matrix(cm_shot_shotscore, list(y_val.action_status_sum.unique()), 'shot_score_cm_xgboost',
                      normalize=False)

xgboost_train_orn = xgboost_orn.predict(x_train)
xgboost_val_orn = xgboost_orn.predict(x_val)
cm_shot_orb = confusion_matrix(xgboost_val_orn, y_val.orb)
plot_confusion_matrix(cm_shot_orb, list(y_val.orb.unique()), 'cm_shot_orb_xg', normalize=False)

xgboost_train_stl = xgboost_stl.predict(x_train)
xgboost_val_stl = xgboost_stl.predict(x_val)
cm_shot_stl = confusion_matrix(xgboost_val_stl, y_val.stl)
plot_confusion_matrix(cm_shot_stl, list(y_val.stl.unique()), 'cm_shot_stl_xg', normalize=False)

xgboost_train_turn = xgboost_turn.predict(x_train)
xgboost_val_turn = xgboost_turn.predict(x_val)
cm_shot_turn = confusion_matrix(xgboost_val_turn, y_val.turn)
plot_confusion_matrix(cm_shot_turn, list(y_val.turn.unique()), 'cm_shot_turn_xg', normalize=False)

########################################################
###########       ALL ACTIONS MODEL ####################
########################################################

# q_for_tbl = constant.query_ml_oloc

q_for_tbl = """select a_.year,
                    a_.game_id,
                    a_.time_marker,
                     case 
                        when a_.quarter = 4 then 'q4' 
                        when a_.quarter = 3 then 'q3' 
                        when a_.quarter = 2 then 'q2' 
                        else 'q1'
                        end AS quarter,
                    a_.q_minutes,
                    a_.team,
                    case
                        when a_.five_on_court = 0 then 1 
                        else a_.five_on_court
                    end as oloc,
                    a_.score_dif,
                    a_.shot_score,
                    a_.shot_miss,
                    a_.foul_made,
                    a_.foul_gain,
                    a_.home_away,
                    case when sub_after_shot_by_team = 1 then 'sub_after_shot_1' else 'sub_after_shot_0' end as sub_after_shot_by_team ,
                    case when sub_after_miss_by_team = 1 then 'sub_after_miss_1' else 'sub_after_miss_0' end as sub_after_miss_by_team ,
                    case when sub_after_foul = 1 then 'sub_after_foul_1' else 'sub_after_foul_0' end as sub_after_foul ,
                    cv_score_dif,
                    abs(cv_score_dif) as abs_cv_score_dif,
                    case
                      when (abs(cv_score_dif) >= 5) then 'Very high'
                      when (abs(cv_score_dif) < 5 and abs(cv_score_dif) >= 2.5) then 'High'
                      when (abs(cv_score_dif) < 2.5 and abs(cv_score_dif) >= 1.5) then 'Mid'
                      when (abs(cv_score_dif) < 1.5 and abs(cv_score_dif) >= 0.5) then 'Low'
                      else 'Very low'
                    end as group_cv
                    , case 
                        when cluster = 3 then 'high_team' 
                        when cluster = 2 then 'med_team' 
                        else 'low_team'
                        end as cluster
                    from basket.data_analytics_all_metrics_agg a_
                    left join 
                    (select distinct year, game_id, time_marker, team,
                    case 
                        when (action in ('Entra a Pista','Sale a Banquillo') and action_p_1_by_team in ('Mate','Canasta de 3','Canasta de 2','Canasta de 1')) then 1
                        else 0 	
                    end sub_after_shot_by_team,
                    case 
                        when (action in ('Entra a Pista','Sale a Banquillo') and action_p_1_by_team in ('Intento fallado de 1', 'Intento fallado de 2','Intento fallado de 3')) then 1
                        else 0 	
                    end sub_after_miss_by_team,
                    case 
                        when (action in ('Entra a Pista','Sale a Banquillo') and action_p_1_by_team in ('Falta Personal','Falta recibida')) then 1
                        else 0 	
                    end sub_after_foul
                    from(
                        select year, game_id, play_number, team , time_marker, action
                        , lag(action,-1) over(partition by year, game_id, team order by play_number asc) action_p_1_by_team
                        , lag(action,-1) over(partition by year, game_id order by play_number asc) action_p_1
                        from stg.all_data_info
                        -- where year = 2003
                        where team !='NA'
                        order by 1,2,3 asc
                    ) as d_
                    order by 1,2,3) as b_
                    on a_.year = b_.year
                    and a_.game_id = b_.game_id
                    and a_.time_marker = b_.time_marker

                    left join 
                    basket.score_dif_cv_per_game c_
                    on a_.year = c_.year
                    and a_.game_id = c_.game_id

                    left join 
                    basket.team_info d_
                    on a_.team = d_.team

                    where score_dif is not null
                    and a_.year = 2003
                    --limit 10000
                    """

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
x_train, x_val, y_train, y_val = train_test_split(train.loc[:, x_col], train.loc[:, y_col], train_size=0.8, random_state=42)

## SHOT SCORE
dt = load("ml_models/train_dt_oloc.joblib")
knn = load("ml_models/train_knn_oloc.joblib")
rnf = load("ml_models/train_rnf_oloc.joblib")
xgboost = load("ml_models/train_xgboost_oloc.joblib")

# TRAIN
dt_train = dt.predict(x_train)
knn_train = knn.predict(x_train)
rnf_train = rnf.predict(x_train)
xgboost_train = xgboost.predict(x_train)

# VAL
dt_val = dt.predict(x_val)
knn_val = knn.predict(x_val)
rnf_val = rnf.predict(x_val)
xgboost_val = xgboost.predict(x_val)

evaluate_mul_class(dt_train, y_train.oloc, 'dt_train_oloc')
evaluate_mul_class(knn_train, y_train.oloc, 'knn_train_oloc')
evaluate_mul_class(rnf_train, y_train.oloc, 'rnf_train_oloc')
evaluate_mul_class(xgboost_train, y_train.oloc, 'xgboost_train_oloc')

evaluate_mul_class(dt_val, y_val.oloc, 'dt_val_oloc')
evaluate_mul_class(knn_val, y_val.oloc, 'knn_val_oloc')
evaluate_mul_class(rnf_val, y_val.oloc, 'rnf_val_oloc')
evaluate_mul_class(xgboost_val, y_val.oloc, 'xgboost_val_oloc')

xgb_model = xgb.XGBClassifier(objective=" multi:softmax"
                              , verbosity=2, eta = 0.01
                              , gamma = 3
                              , random_state=42
                              , max_depth = 10
                              , subsample  = 0.85
                              , num_class = len(y_train.oloc.unique())
                              , alpha = 2
                              , eval_metric = 'merror'
                              , predictor = "gpu_predictor"
                              )

le = LabelEncoder()
y_train = pd.Series(le.fit_transform(y_train))
y_val = pd.Series(le.fit_transform(y_val))

t0 = time.time()
xgb_model.fit(x_train, y_train)
t1 = time.time()
print("DONE PREPROCESSING IN ", round((t1 - t0)), " SEC")


xgboost_train = xgb_model.predict(x_train)
xgboost_val = xgb_model.predict(x_val)

evaluate_mul_class(xgboost_train, y_train, 'xgboost_train_oloc')
evaluate_mul_class(xgboost_val, y_val, 'xgboost_val_oloc')