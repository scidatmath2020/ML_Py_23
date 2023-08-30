# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:18:45 2023

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

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")


#%%

mi_data = pd.read_csv("CC_GENERAL.csv")

mi_data.shape

mi_data.dtypes

#%%

customer_ids = mi_data.CUST_ID
mi_data = mi_data >> select(-_.CUST_ID)

mi_data.columns[mi_data.isnull().any()]

mi_data[mi_data.isnull().any(axis=1)]

mi_data = mi_data.fillna(0)

#%%
'''
Hemos aprendido el algoritmo DBSCAN. Lo bueno de usar este algoritmo para 
detección de anomalías es que no asigna un cluster a todos los puntos, sino 
aquellos puntos que están más separados del resto se etiquetan automáticamente 
como valores extremos.
'''

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

'''Al usar un algoritmo de clustering basado en densidad tenemos que 
estandarizar los datos'''

mi_data_normalizado = pd.DataFrame(StandardScaler().fit_transform(mi_data))

clusterer = DBSCAN()
cluster_labels = clusterer.fit_predict(mi_data_normalizado)
pd.Series(cluster_labels).value_counts()

#%%
'''
Vemos que por defecto DBSCAN produce demasiados elementos anómalos del total de
9000 transacciones. Podemos usar el coeficiente de silueta (silhouette_score) 
para ver como separa el algoritmo a los buenos clientes de los (potencialmente)
malos.
'''

from sklearn.metrics import silhouette_score
silhouette_score(mi_data_normalizado, cluster_labels)

DBSCAN().get_params()

#%%
'''
Podemos hacer una búsqueda aleatoria para optimizar dicho resultado.
'''

from scipy.stats import randint as sp_randint
from scipy.stats import uniform 


distribucion_parametros = {
    "eps": uniform(0,5),
    "min_samples": sp_randint(2, 20),
    "p": sp_randint(1, 3),
}

distribucion_parametros
#%%
'''
Un problema que hay con el algoritmo HDBSCAN (o DBSCAN) es que no tiene el 
método predict, por lo tanto no podemos usar el método de busqueda aleatoria 
de scikit-learn.

Sin embargo, podemos desarrollar nuestro propio método de búsqueda con 
ParameterSampler, que es lo que usa scikit-learn para tomar muestras del 
diccionario de búsqueda de hiperparámetros.

Este paso tarda tiempo en ejecutarse
'''

from sklearn.model_selection import ParameterSampler

n_muestras = 10 # probamos 5 combinaciones de hiperparámetros
n_iteraciones = 3 #para validar, vamos a entrenar para cada selección de hiperparámetros en 3 muestras distintas
pct_muestra = 0.7 # usamos el 70% de los datos para entrenar el modelo en cada iteracion
resultados_busqueda = []
lista_parametros = list(ParameterSampler(distribucion_parametros, n_iter=n_muestras))

for param in lista_parametros:
    for iteration in range(n_iteraciones):
        param_resultados = []
        muestra = mi_data_normalizado.sample(frac=pct_muestra)
        etiquetas_clusters = DBSCAN(n_jobs=-1, **param).fit_predict(muestra)
        try:
            param_resultados.append(silhouette_score(muestra, etiquetas_clusters))
        except ValueError: # a veces silhouette_score falla en los casos en los que solo hay 1 cluster
            pass
    puntuacion_media = np.mean(param_resultados)
    resultados_busqueda.append([puntuacion_media, param])
    
sorted(resultados_busqueda, key=lambda x: x[0], reverse=True)[:5]

#%%

mejores_params = {'eps': 4.9856603649238247, 'min_samples':6, 'p': 1}

clusterer = DBSCAN(n_jobs=-1, **mejores_params)

etiquetas_cluster = clusterer.fit_predict(mi_data_normalizado)

pd.Series(etiquetas_cluster).value_counts()

#%%

def resumen_cluster(cluster_id):
    cluster = mi_data[etiquetas_cluster==cluster_id]
    resumen_cluster = cluster.mean().to_dict()
    resumen_cluster["cluster_id"] = cluster_id
    return resumen_cluster

def comparar_clusters(*cluster_ids):
    resumenes = []
    for cluster_id in cluster_ids:
        resumenes.append(resumen_cluster(cluster_id))
    return pd.DataFrame(resumenes).set_index("cluster_id").T

comparar_clusters(0,-1)
