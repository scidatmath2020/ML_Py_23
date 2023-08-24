# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:47:17 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import *
from plotnine import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")

#%%

#mi_data = pd.read_csv("datos_separados.csv")
mi_data = pd.read_csv("datos_circulares.csv")

(ggplot(data = mi_data) +
     geom_point(mapping=aes(x="columna1",y="columna2"),alpha=0.2) 
)


#%%

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

#%%
modelos = KMeans()

visualizer = KElbowVisualizer(modelos, k=(2,10),metric = "silhouette")
visualizer.fit(mi_data)
visualizer.show()

#%%

k_medias = KMeans(n_clusters = 2 ,init='k-means++', n_init = 10)
k_medias.fit(mi_data)
etiquetas_k_medias = k_medias.labels_ 
silhouette_score(mi_data,etiquetas_k_medias)

(mi_data >> mutate(Cluster_km = etiquetas_k_medias) >>
     ggplot() +
          geom_point(mapping=aes(x="columna1",y="columna2",color = "Cluster_km.astype(str)"),alpha=0.2) 
 
 )

#%%

dbscan = DBSCAN(eps=0.03, min_samples=20)
etiquetas_dbscan = dbscan.fit(mi_data).labels_
silhouette_score(mi_data,etiquetas_dbscan)

np.unique(etiquetas_dbscan)

(mi_data >> mutate(Cluster_dbs = etiquetas_dbscan) >>
     ggplot() +
          geom_point(mapping=aes(x="columna1",y="columna2",color = "Cluster_dbs.astype(str)"),alpha=0.2) 
 
 )

#%%

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#%%

def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):

    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X) 
                                       
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()


k = 2 * mi_data.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(X=mi_data, k=k)
