from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sc
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import utils.postgres_config as psql
import math
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

q = """
with tem as(
select year, game_id, team
, sum(case
    when action = 'Canasta de 3' then 3 
	when action in ('Canasta de 2', 'Mate') then 2
    when action = 'Canasta de 1' then 1
	else 0
end) points_score
, sum(case
    when action = 'Canasta de 3' then 1
   else 0
end) as PTS3
, sum(case
    when action in ('Canasta de 2', 'Mate') then 1
	else 0
end) as PTS2
, sum(case
    when action = 'Canasta de 1' then 1
	else 0
end) as PTS1
, sum(case
    when action = 'Intento fallado de 3' then 1 
   else 0
end) as PTA3
, sum(case
    when action in ('Intento fallado de 2') then 1
	else 0
end) as PTA2
, sum(case
    when action = 'Intento fallado de 1' then 1
	else 0
end) as PTA1
, sum(case
    when action = 'Asistencia' then 1
	else 0
end) as AST
, sum(case
    when action = 'Recuperación' then 1
	else 0
end) as STL
, sum(case
    when action = 'Tapón' then 1
	else 0
end) as BLK
, sum(case
    when action = 'Rebote Defensivo' then 1
	else 0
end) as DRB
, sum(case
    when action = 'Rebote Ofensivo' then 1
	else 0
end) as ORB
, sum(case
    when action = 'Pérdida' then 1
	else 0
end) as TRN
from stg.all_data_info
group by 1,2,3
order by 1,2,3 asc
) select team, year
,avg(points_score) as points_score
,avg(PTS3) as PTS3
,avg(PTS2) as PTS2
,avg(PTS1) as PTS1
,avg(PTA3) as PTA3
,avg(PTA2) as PTA2
,avg(PTA1) as PTA1
,avg(AST) as AST
,avg(STL) as STL
,avg(BLK) as BLK
,avg(DRB) as DRB
,avg(ORB) as ORB
,avg(TRN) as TRN
from tem
where team != 'NA'
group by 1,2
"""

q2 = """
with tem as(
select year, game_id, team
, sum(case
    when action = 'Canasta de 3' then 3 
	when action in ('Canasta de 2', 'Mate') then 2
    when action = 'Canasta de 1' then 1
	else 0
end) points_score
, sum(case
    when action = 'Canasta de 3' then 1
   else 0
end) as PTS3
, sum(case
    when action in ('Canasta de 2', 'Mate') then 1
	else 0
end) as PTS2
, sum(case
    when action = 'Canasta de 1' then 1
	else 0
end) as PTS1
, sum(case
    when action = 'Intento fallado de 3' then 1 
   else 0
end) as PTA3
, sum(case
    when action in ('Intento fallado de 2') then 1
	else 0
end) as PTA2
, sum(case
    when action = 'Intento fallado de 1' then 1
	else 0
end) as PTA1
, sum(case
    when action = 'Asistencia' then 1
	else 0
end) as AST
, sum(case
    when action = 'Recuperación' then 1
	else 0
end) as STL
, sum(case
    when action = 'Tapón' then 1
	else 0
end) as BLK
, sum(case
    when action = 'Rebote Defensivo' then 1
	else 0
end) as DRB
, sum(case
    when action = 'Rebote Ofensivo' then 1
	else 0
end) as ORB
, sum(case
    when action = 'Pérdida' then 1
	else 0
end) as TRN
from stg.all_data_info
group by 1,2,3
order by 1,2,3 asc
) select team
,avg(points_score) as points_score
,avg(PTS3) as PTS3
,avg(PTS2) as PTS2
,avg(PTS1) as PTS1
,avg(PTA3) as PTA3
,avg(PTA2) as PTA2
,avg(PTA1) as PTA1
,avg(AST) as AST
,avg(STL) as STL
,avg(BLK) as BLK
,avg(DRB) as DRB
,avg(ORB) as ORB
,avg(TRN) as TRN
from tem
where team != 'NA'
group by 1
"""

def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return sc(X, cluster_labels)

def plot_dendrogram(model,df,  **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    print(linkage_matrix)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,labels=df.team, **kwargs)

    plt.axhline(y=9, c='grey', lw=1, linestyle='dashed')


con = psql.PostgressCon()
con.connect()
res = con.execute_query_with_headers(q)
df = pd.DataFrame(data = res[0], columns=res[1])
data = df.iloc[:,2::]

res = con.execute_query_with_headers(q2)
df_2 = pd.DataFrame(data = res[0], columns=res[1])
data2 = df_2.iloc[:,1::]


scalar = StandardScaler().fit(data)
x = scalar.fit_transform(data)
scalar = StandardScaler().fit(data2)
x2 = scalar.fit_transform(data2)

knn = KMeans()

modelGs = GridSearchCV(knn
                       , {'n_clusters': [1, 2, 3, 4,5,6,7]}
                       , error_score='raise'
                       , scoring= cv_silhouette_scorer)
modelGs.fit(x)


knn = AgglomerativeClustering(n_clusters = 3, compute_distances = True)
knn.fit(x)
plot_dendrogram(knn, truncate_mode="level", p=3)

knn.fit_predict(x2)
plot_dendrogram(knn, df_2, truncate_mode="level", p=3)



df.head(1).to_string()

df_2['cluster'] = list(knn.fit_predict(x2))


pca = PCA(n_components=4)
pca.fit(x2.transpose())
print(pca.explained_variance_ratio_)
plt.switch_backend('TkAgg')
features = list(df_2.columns[1:-1])
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

plt.scatter(pca.components_[0], pca.components_[1], c=list(knn.fit_predict(x2)), cmap=plt.cm.nipy_spectral, s=50)
plt.title("PCA 2 components projection (with clustering partition)", fontsize=15)
plt.xlabel("1st PC", fontsize=15)
plt.ylabel("2st PC", fontsize=15)
num = 0
for x,y in zip(pca.components_[0],pca.components_[1]):

    label = df_2.iloc[num,0]

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 fontsize = 15,
                 ha='center') # horizontal alignment can be left, right or center
    num+=1

for i, feature in enumerate(features):
    plt.arrow(x=0, y=0,
        dx=loadings[i, 0],
        dy=loadings[i, 1],
        color = 'lightblue',
        linestyle = '--',
        linewidth = 2,
        head_width = 0.03
    )
    plt.text(
        x=loadings[i, 0],
        y=loadings[i, 1],
        s=feature,
        fontsize=14,
        color = 'red'

    )
plt.show()
# plt.savefig("pca.png")

q_team = "select team, team_name, stadium from basket.team_info"
res_team = con.execute_query_with_headers(q_team)
df_team = pd.DataFrame(data = res_team[0], columns=res_team[1])



df_2['cluster'] = df_2['cluster'] +1

df_team = df_team.merge(df_2.loc[:,['team', 'cluster']],how = 'left', on= 'team')
df_team.drop('cluster_x', inplace = True, axis=1)
df_team.columns = ['team', 'team_name', 'stadium', 'cluster']
con.table_from_df(df_team,"basket", "team_info_2")







