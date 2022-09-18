import pandas as pd
import utils.postgres_config as psql
import utils.constant_class as const
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def plot_pca_by_group(pca_player, scatter_colors):
    fig, axes = plt.subplots(5, 5, figsize=(10,10))

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if row == col:
                ax.tick_params(
                    axis='both', which='both',
                    bottom='off', top='off',
                    labelbottom='off',
                    left='off', right='off',
                    labelleft='off'
                )
                ax.text(0.5, 0.5, pca_str[row], horizontalalignment='center')
            else:
                ax.scatter(pca_player[:, row], pca_player[:, col], c=scatter_colors, s=3, )
    fig.tight_layout()
    plt.show()

# Data process
dbConn_obj = psql.PostgressCon()

q = """select player_name, avg(pts) as pts, avg(tps_1) as tps_1, avg(tps_2) as tps_2, avg(tps_3) as tps_3,
       avg(fail_1) as fail_1, avg(fail_2) as fail_2, avg(fail_3) as fail_3, avg(atm_1) as atm_1, avg(atm_2) as atm_2, avg(atm_3) as atm_3, avg(turnover) as turnover,
       avg(blk) as blk, avg(drb) as drb, avg(stl) as stl, max(position) as position, max(tempo) as tempo, max(age) as age
from (
(select *
from basket.player_game_performance  
where year = 2003
) as a_
	left join
(select player, team, position, "temp." as tempo, age
 from basket.player_year_group 
 where year = 2003) as b_
 on a_.player_name = b_.player
	)
	 group by 1"""

data, col_name = dbConn_obj.execute_query_with_headers(q)
df = pd.DataFrame(data = data, columns = col_name)
x = df.iloc[:, 1:15]

# Fitting
k = [k_ for k_ in range(2,20)]
k_kmeans = [ KMeans(n_clusters=model).fit(x) for model in k]
k_str = [ "model_" + str(k_) for k_ in k]

for col_k, model in zip(k_str, k_kmeans):
    df[col_k] = model.labels_

# Plotting
import matplotlib.colors as mcolors
temp = mcolors.CSS4_COLORS
colors_number = list(mcolors.CSS4_COLORS.keys())[3::]

pca_player = PCA(5).fit_transform(x)
pca_str = [ "pc_" + str(i) for i in range(1,6)]
pca_df = pd.DataFrame(pca_player, columns=pca_str)
scatter_matrix(pca_df, alpha=0.2, figsize=(6, 6), diagonal="kde")
col_1 = {k_: colors_number[k_] for k_ in range(0,4)}
group_col = list(df['model_3'].map(col_1))
plot_pca_by_group(pca_player, group_col)

