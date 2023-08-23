# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:51:43 2023

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

mi_data = pd.read_csv("datos_iris.csv")
mi_data = mi_data >> select(_.startswith("Sepal")) 

#%%

(ggplot(data = mi_data) +
    geom_point(mapping=aes(x="Sepal_Length",y="Sepal_Width"))
)

#%%

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

mi_data = pd.read_csv("datos_iris.csv")
mi_data = mi_data >> select(_.startswith("Sepal")) 

escalador = preprocessing.normalize(mi_data)
mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=mi_data.index, 
                                      columns=mi_data.columns)

k_medias = KMeans(n_clusters = 2 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
k_medias.fit(mi_data_normalizado_df)
Etiquetas = k_medias.labels_

#%%

silhouette_score(mi_data_normalizado_df,Etiquetas)

#%%

(mi_data >> mutate(siluetas = silhouette_samples(mi_data_normalizado_df,Etiquetas),
                  etiquetas = Etiquetas.astype(str)) >>
    ggplot() +
        geom_point(mapping=aes(x="Sepal_Length",y="Sepal_Width",color = "siluetas",shape="etiquetas"))
)

#%%

def constructor_clusters(data,k):
    escalador = preprocessing.normalize(data)
    mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=data.index, 
                                      columns=data.columns)

    k_medias = KMeans(n_clusters = k ,init='k-means++')
    k_medias.fit(mi_data_normalizado_df)
    Etiquetas = k_medias.labels_
    silueta = silhouette_score(mi_data_normalizado_df,Etiquetas)
    cal_har = calinski_harabasz_score(mi_data_normalizado_df,Etiquetas)
    
    return k, Etiquetas, silueta, cal_har 

#%%

constructor_clusters(mi_data,4)

#%%

modelos_kmedias = [constructor_clusters(mi_data,k) for k in range(2,10)]

#%%

resultados = pd.DataFrame([(x[0],x[2],x[3]) for x in modelos_kmedias],
             columns = ["k","silueta","calinski_harabasz"])

#%%

(ggplot(data = resultados) +
    geom_point(mapping = aes(x="k",y="silueta"),color = "red") +
    geom_line(mapping = aes(x="k",y="silueta"),color = "red") 
)

#%%

modelos = KMeans()

visualizer = KElbowVisualizer(modelos, k=(2,10),metric = "silhouette")
visualizer.fit(mi_data_normalizado_df)
visualizer.show()

#%%

modelos = KMeans()

visualizer = KElbowVisualizer(modelos, k=(2,10),metric = "calinski_harabasz")
visualizer.fit(mi_data_normalizado_df)
visualizer.show()
