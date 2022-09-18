# -*- coding: utf-8 -*-
"""rtdl.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Yura52/rtdl/blob/main/examples/rtdl.ipynb
"""

# Requirements:
from typing import Any, Dict

import numpy as np
import pandas as pd
import utils.postgres_config as psql
from utils.constant_class import constant
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero

device = torch.device('cpu')
# Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)

"""### Data"""

# !!! NOTE !!! The dataset splits, preprocessing and other details are
# significantly different from those used in the
# paper "Revisiting Deep Learning Models for Tabular Data",
# so the results will be different from the reported in the paper.

task_type = 'multiclass'

# dataset = sklearn.datasets.fetch_covtype()
# task_type = 'multiclass'

assert task_type in ['binclass', 'multiclass', 'regression']


q_for_tbl = constant.query_ml_oloc

dbConn_obj = psql.PostgressCon()
df = dbConn_obj.execute_query_with_headers(q_for_tbl)
df = pd.DataFrame(data=df[0], columns=df[1])

df.time_marker = df.time_marker.astype("int32")
df.cv_score_dif = df.cv_score_dif.astype("float32")
df.abs_cv_score_dif = df.abs_cv_score_dif.astype("float32")
df.score_dif = df.score_dif.astype("int32")
df.q_minutes = df.q_minutes.astype("int32")
df.shot_score = df.shot_score.astype("int32")
df.shot_miss = df.shot_miss.astype("int32")
df.foul_made = df.foul_made.astype("int32")
df.foul_gain = df.foul_gain.astype("int32")

numeric_list = ['time_marker',
                'score_dif',
                'q_minutes',
                'cv_score_dif',
                'abs_cv_score_dif',
                'shot_score',
                'shot_miss',
                'foul_made',
                'foul_gain'
                ]

df.sub_after_shot_by_team = df.sub_after_shot_by_team.astype("string")
df.sub_after_miss_by_team = df.sub_after_miss_by_team.astype("string")
df.sub_after_foul = df.sub_after_foul.astype("string")
df.group_cv = df.group_cv.astype("string")
df.cluster = df.cluster.astype("string")

enc_team = pd.get_dummies(df.team)
enc_ha = pd.get_dummies(df.home_away)
enc_q_minutes = pd.get_dummies(df.q_minutes)
enc_quarter = pd.get_dummies(df.quarter)
enc_sub_after_shot_by_team = pd.get_dummies(df.sub_after_shot_by_team)
enc_sub_after_miss_by_team = pd.get_dummies(df.sub_after_miss_by_team)
enc_sub_after_foul = pd.get_dummies(df.sub_after_foul)
enc_group_cv = pd.get_dummies(df.group_cv)
enc_cluster = pd.get_dummies(df.cluster)

df = pd.concat([df, enc_team], axis=1)
df = pd.concat([df, enc_ha], axis=1)
df = pd.concat([df, enc_quarter], axis=1)
df = pd.concat([df, enc_sub_after_shot_by_team], axis=1)
df = pd.concat([df, enc_sub_after_miss_by_team], axis=1)
df = pd.concat([df, enc_sub_after_foul], axis=1)
df = pd.concat([df, enc_group_cv], axis=1)
df = pd.concat([df, enc_cluster], axis=1)

df.oloc = df.oloc.astype("int32")
df.oloc = df.oloc-1

x_col = [ 'time_marker', 'q_minutes', 'score_dif', 'shot_score', 'shot_miss',
          'foul_made', 'foul_gain','cv_score_dif',
           'abs_cv_score_dif', 'ALI', 'BAR', 'BAS', 'BRE', 'CLA', 'EST', 'FUE', 'GIR', 'JOV',  'LLE',
           'MAD', 'MAN', 'MUR', 'PAM', 'RON', 'SEV', 'TNF', 'VAL',
           'away_team', 'home_team', 'q1', 'q2', 'q3', 'q4', 'sub_after_shot_0',
           'sub_after_shot_1', 'sub_after_miss_0', 'sub_after_miss_1',
           'sub_after_foul_0', 'sub_after_foul_1', 'High', 'Low', 'Mid',
           'Very high', 'Very low', 'high_team', 'low_team', 'med_team']
df = df.loc[:, x_col+['oloc']]


X_all = df[x_col].astype('float32')
y_all = df['oloc'].astype('float32' if task_type == 'regression' else 'int64')
if task_type != 'regression':
    y_all = sklearn.preprocessing.LabelEncoder().fit_transform(y_all).astype('int64')
n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None

X = {}
y = {}
X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
    X_all.loc[:,x_col], y_all, train_size=0.8
)
X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
    X['train'], y['train'], train_size=0.8
)

# not the best way to preprocess features, but enough for the demonstration
preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
X = {
    k: torch.tensor(preprocess.fit_transform(v), device=device)
    for k, v in X.items()
}
y = {k: torch.tensor(v, device=device) for k, v in y.items()}

# !!! CRUCIAL for neural networks when solving regression problems !!!
if task_type == 'regression':
    y_mean = y['train'].mean().item()
    y_std = y['train'].std().item()
    y = {k: (v - y_mean) / y_std for k, v in y.items()}
else:
    y_std = y_mean = None

if task_type != 'multiclass':
    y = {k: v.float() for k, v in y.items()}

"""### Model
Carefully read the comments and uncomment the code for the model you want to test.
"""

d_out = n_classes or 1

# model = rtdl.MLP.make_baseline(
#     d_in=X_all.shape[1],
#     d_layers=[128, 256, 128],
#     dropout=0.1,
#     d_out=d_out,
# )
# lr = 0.001
# weight_decay = 0.0

# model = rtdl.ResNet.make_baseline(
#     d_in=X_all.shape[1],
#     n_blocks=5,
#     d_main=5,
#     d_hidden=5,
#     dropout_first=0.25,
#     dropout_second=0.1,
#     d_out=d_out,
# )
# lr = 0.1
# weight_decay = 0.0

grid = {
    'd_token': [96, 128, 192, 256],
    'attention_dropout': [0.1, 0.15, 0.2, 0.25],
    'ffn_dropout': [0.0, 0.05, 0.1, 0.15],
}
arch_subconfig = {k: v[4 - 1] for k, v in grid.items()}  # type: ignore
baseline_subconfig = rtdl.FTTransformer.get_baseline_transformer_subconfig()

model = rtdl.FTTransformer.make_baseline(
    n_num_features=X_all.shape[1],
    n_blocks = 4,
    cat_cardinalities=None,
    residual_dropout=0.0,  # it makes the model faster and does NOT affect its output
    d_out=d_out,
    ffn_d_hidden=6,
    **arch_subconfig

)

# === ABOUT CATEGORICAL FEATURES ===
# IF you use MLP, ResNet or any other simple feed-forward model (NOT transformer-based model)
# AND there are categorical features
# THEN you have to implement a wrapper that handles categorical features.
# The example below demonstrates how it can be achieved using rtdl.CategoricalFeatureTokenizer.
# ==================================
# 1. When you have both numerical and categorical features, you should prepare you data like this:
#    (X_num<float32>, X_cat<int64>) instead of X<float32>
#    Each column in X_cat should contain values within the range from 0 to <(the number of unique values in column) - 1>;
#    use sklean.preprocessing.OrdinalEncoder to achieve this;
# 2. Prepare a list of so called "cardinalities":
#    cardinalities[i] = <the number of unique values of the i-th categorical feature>
# 3. See the commented example below and adapt it for your needs.
#
# class Model(nn.Module):
#     def __init__(
#         self,
#         n_num_features: int,
#         cat_tokenizer: rtdl.CategoricalFeatureTokenizer,
#         mlp_kwargs: Dict[str, Any],
#     ):
#         super().__init__()
#         self.cat_tokenizer = cat_tokenizer
#         self.model = rtdl.MLP.make_baseline(
#             d_in=n_num_features + cat_tokenizer.n_tokens * cat_tokenizer.d_token,
#             **mlp_kwargs,
#         )
#
#     def forward(self, x_num, x_cat):
#         return self.model(
#             torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
#         )
#
# model = Model(
#     # `None` means "Do not transform numerical features"
#     # `d_token` is the size of embedding for ONE categorical feature
#     X_num_all.shape[1],
#     rtdl.CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform'),
#     mlp_kwargs,
# )
# Then the model should be used as `model(x_num, x_cat)` instead of of `model(x)`.

model.to(device)
optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=0.9, weight_decay=0.8)
)
loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
)

"""### Training"""

def apply_model(x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(part):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        prediction.append(apply_model(batch))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return score


# Create a dataloader for batches of indices
# Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
batch_size = 256
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=100)

print(f'Test score before training: {evaluate("test"):.4f}')

n_epochs = 2
report_frequency = len(X['train']) // batch_size // 5
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

    val_score = evaluate('val')
    test_score = evaluate('test')
    print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    progress.update((-1 if task_type == 'regression' else 1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
    print()
    if progress.fail:
        break

model.eval()
prediction = []
for batch in zero.iter_batches(X['test'][0:2000], 1024):
    prediction.append(apply_model(batch))
tem = torch.cat(prediction).squeeze(1).cpu()
tem = tem.detach().numpy()
target = y['test'][0:2000].cpu().numpy()
pred = tem.argmax(axis=1)
#
(pred-target).mean()

temp = pd.DataFrame(columns=['y_pred','y'])
temp['y_pred'] = pred
temp['y'] = target

temp.to_csv('FTTransformer_results.csv', index=False)